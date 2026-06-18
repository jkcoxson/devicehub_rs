// An RFB (VNC) server: a second consumer of the UI's `FrameSlot` and a second
// producer into its `InputSink`. RFB 3.8, Raw encoding only.

use std::sync::{Arc, Mutex};
use std::time::Duration;

use flate2::{Compress, Compression, FlushCompress};
use tokio::io::{AsyncReadExt, AsyncWriteExt};
use tokio::net::tcp::{OwnedReadHalf, OwnedWriteHalf};
use tokio::net::{TcpListener, TcpStream};
use tokio::task::JoinHandle;

use crate::protocol::{
    Frame, FrameSlot, InputCmd, InputSink, KeyMods, OrientationSlot, VncControl, norm,
    unrotate_norm,
};

/// Minimum interval between framebuffer pushes (~30 fps).
const FRAME_INTERVAL: Duration = Duration::from_millis(33);

/// Whether to offer the `Zlib` encoding (RFB 6). Disabled: macOS Screen Sharing
/// advertises it but its plain-Zlib decoder mishandles the stream (intermittent
/// colour bands / flashes), while our diffed Raw rectangles decode cleanly. The
/// proper compressed path for that client is ZRLE (encoding 16), not Zlib.
const ALLOW_ZLIB: bool = false;

/// Framebuffer size advertised in `ServerInit` before any frame arrives; the
/// real size follows via a desktop-size update.
const DEFAULT_FB: (u16, u16) = (390, 844);

/// Compute the 16-byte RFB VNC-auth response for `password` over `challenge`.
///
/// The password is truncated/zero-padded to 8 bytes, each byte is bit-reversed
/// (a VNC quirk), and the result is the DES key to ECB-encrypt the challenge.
fn vnc_auth_response(password: &str, challenge: &[u8; 16]) -> [u8; 16] {
    use des::Des;
    use des::cipher::generic_array::GenericArray;
    use des::cipher::{BlockEncrypt, KeyInit};

    let pw = password.as_bytes();
    let mut key = [0u8; 8];
    for (i, slot) in key.iter_mut().enumerate() {
        *slot = pw.get(i).copied().unwrap_or(0).reverse_bits();
    }

    let cipher = Des::new(GenericArray::from_slice(&key));
    let mut out = [0u8; 16];
    for chunk in 0..2 {
        let range = chunk * 8..chunk * 8 + 8;
        let mut block = *GenericArray::from_slice(&challenge[range.clone()]);
        cipher.encrypt_block(&mut block);
        out[range].copy_from_slice(&block);
    }
    out
}

/// 16 bytes of best-effort (non-cryptographic) randomness for the auth
/// challenge, so it isn't a fixed replayable constant. splitmix64 off the clock.
fn random_challenge() -> [u8; 16] {
    let mut state = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_nanos() as u64)
        .unwrap_or(0x9e3779b97f4a7c15);
    let mut next = || {
        state = state.wrapping_add(0x9e3779b97f4a7c15);
        let mut z = state;
        z = (z ^ (z >> 30)).wrapping_mul(0xbf58476d1ce4e5b9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94d049bb133111eb);
        z ^ (z >> 31)
    };
    let mut out = [0u8; 16];
    out[..8].copy_from_slice(&next().to_le_bytes());
    out[8..].copy_from_slice(&next().to_le_bytes());
    out
}

/// Aborts the task it holds when dropped — used to tear down a client's
/// message-reader task when its connection handler returns.
struct AbortOnDrop(JoinHandle<()>);

impl Drop for AbortOnDrop {
    fn drop(&mut self) {
        self.0.abort();
    }
}

/// A running VNC listener: the accept-loop task plus its per-client tasks.
struct Running {
    accept: JoinHandle<()>,
    clients: Arc<Mutex<Vec<JoinHandle<()>>>>,
}

impl Running {
    /// Abort the accept loop and every live client connection.
    fn shutdown(self) {
        self.accept.abort();
        for handle in self.clients.lock().unwrap().drain(..) {
            handle.abort();
        }
    }
}

/// Supervise the VNC server, binding/unbinding the listener to match the
/// `control`'s desired state. Reconciles when woken via `control.wake()`.
pub async fn supervise(
    control: VncControl,
    frames: FrameSlot,
    input: InputSink,
    orientation: OrientationSlot,
) {
    let wake = control.wake();
    let mut running: Option<Running> = None;
    loop {
        // Arm the wake-up *before* reconciling so a toggle during an `.await`
        // below isn't lost (Notify stores one permit).
        let notified = wake.notified();

        match (control.enabled(), running.is_some()) {
            (true, false) => {
                let addr = control.addr();
                match TcpListener::bind(&addr).await {
                    Ok(listener) => {
                        let bound = listener
                            .local_addr()
                            .map(|a| a.to_string())
                            .unwrap_or_else(|_| addr.clone());
                        let auth = if control.password().is_empty() {
                            "no auth - keep on a trusted network"
                        } else {
                            "password required"
                        };
                        tracing::info!("VNC server listening on {bound} ({auth})");
                        control.set_status(format!("listening on {bound}"));
                        let clients = Arc::new(Mutex::new(Vec::new()));
                        let accept = tokio::spawn(accept_loop(
                            listener,
                            frames.clone(),
                            input.clone(),
                            orientation.clone(),
                            control.clone(),
                            clients.clone(),
                        ));
                        running = Some(Running { accept, clients });
                    }
                    Err(e) => {
                        tracing::warn!("VNC: failed to bind {addr}: {e}");
                        control.set_status(format!("error: {e}"));
                        control.set_enabled(false);
                    }
                }
            }
            (false, true) => {
                if let Some(r) = running.take() {
                    r.shutdown();
                }
                control.set_clients(0);
                control.set_status("stopped");
                tracing::info!("VNC server stopped");
            }
            _ => {}
        }

        notified.await;
    }
}

/// Accept VNC clients forever, serving each on its own task.
async fn accept_loop(
    listener: TcpListener,
    frames: FrameSlot,
    input: InputSink,
    orientation: OrientationSlot,
    control: VncControl,
    client_handles: Arc<Mutex<Vec<JoinHandle<()>>>>,
) {
    loop {
        match listener.accept().await {
            Ok((stream, peer)) => {
                stream.set_nodelay(true).ok();
                let (frames, input, orientation, control) = (
                    frames.clone(),
                    input.clone(),
                    orientation.clone(),
                    control.clone(),
                );
                let handle = tokio::spawn(async move {
                    tracing::info!("VNC client connected: {peer}");
                    control.incr_clients();
                    // Read password per-connection so changes apply without restart.
                    let password = control.password();
                    if let Err(e) =
                        handle_client(stream, frames, input, orientation, password).await
                    {
                        tracing::info!("VNC client {peer} disconnected: {e}");
                    }
                    control.decr_clients();
                });
                let mut handles = client_handles.lock().unwrap();
                handles.retain(|h| !h.is_finished());
                handles.push(handle);
            }
            Err(e) => tracing::warn!("VNC accept error: {e}"),
        }
    }
}

// --- Pixel format -----------------------------------------------------------

/// An RFB pixel format (the 16-byte `PIXEL_FORMAT` structure). We propose a
/// default but honor whatever the client sets with `SetPixelFormat`.
#[derive(Clone, Copy)]
struct PixelFormat {
    bits_per_pixel: u8,
    depth: u8,
    big_endian: bool,
    true_color: bool,
    red_max: u16,
    green_max: u16,
    blue_max: u16,
    red_shift: u8,
    green_shift: u8,
    blue_shift: u8,
}

impl PixelFormat {
    /// Our proposed format: 32 bpp little-endian `0x00RRGGBB` (bytes B,G,R,X).
    fn server_default() -> Self {
        Self {
            bits_per_pixel: 32,
            depth: 24,
            big_endian: false,
            true_color: true,
            red_max: 255,
            green_max: 255,
            blue_max: 255,
            red_shift: 16,
            green_shift: 8,
            blue_shift: 0,
        }
    }

    fn bytes_per_pixel(&self) -> usize {
        (self.bits_per_pixel as usize / 8).max(1)
    }

    /// Encode one RGB triple into this format and append it to `out`.
    fn put(&self, out: &mut Vec<u8>, r: u8, g: u8, b: u8) {
        let scale = |c: u8, max: u16, shift: u8| (c as u32 * max as u32 / 255) << shift;
        let v = scale(r, self.red_max, self.red_shift)
            | scale(g, self.green_max, self.green_shift)
            | scale(b, self.blue_max, self.blue_shift);
        match self.bytes_per_pixel() {
            4 => {
                if self.big_endian {
                    out.extend_from_slice(&v.to_be_bytes());
                } else {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            2 => {
                let v = v as u16;
                if self.big_endian {
                    out.extend_from_slice(&v.to_be_bytes());
                } else {
                    out.extend_from_slice(&v.to_le_bytes());
                }
            }
            _ => out.push(v as u8),
        }
    }
}

async fn read_pixel_format(rd: &mut OwnedReadHalf) -> std::io::Result<PixelFormat> {
    let mut b = [0u8; 16];
    rd.read_exact(&mut b).await?;
    Ok(PixelFormat {
        bits_per_pixel: b[0],
        depth: b[1],
        big_endian: b[2] != 0,
        true_color: b[3] != 0,
        red_max: u16::from_be_bytes([b[4], b[5]]),
        green_max: u16::from_be_bytes([b[6], b[7]]),
        blue_max: u16::from_be_bytes([b[8], b[9]]),
        red_shift: b[10],
        green_shift: b[11],
        blue_shift: b[12],
        // b[13..16] padding
    })
}

async fn write_pixel_format(wr: &mut OwnedWriteHalf, pf: &PixelFormat) -> std::io::Result<()> {
    let mut b = [0u8; 16];
    b[0] = pf.bits_per_pixel;
    b[1] = pf.depth;
    b[2] = pf.big_endian as u8;
    b[3] = pf.true_color as u8;
    b[4..6].copy_from_slice(&pf.red_max.to_be_bytes());
    b[6..8].copy_from_slice(&pf.green_max.to_be_bytes());
    b[8..10].copy_from_slice(&pf.blue_max.to_be_bytes());
    b[10] = pf.red_shift;
    b[11] = pf.green_shift;
    b[12] = pf.blue_shift;
    wr.write_all(&b).await
}

// --- Client messages --------------------------------------------------------

enum ClientMsg {
    SetPixelFormat(PixelFormat),
    SetEncodings(Vec<i32>),
    FbUpdateRequest {
        incremental: bool,
    },
    Key {
        down: bool,
        keysym: u32,
    },
    Pointer {
        mask: u8,
        x: u16,
        y: u16,
    },
    /// Client clipboard paste - consumed but not yet forwarded to the device.
    CutText,
}

/// Read one client->server message. Errors (closing the connection) on EOF or an
/// unrecognized message type, whose length we can't know to skip.
async fn read_client_message(rd: &mut OwnedReadHalf) -> std::io::Result<ClientMsg> {
    let kind = rd.read_u8().await?;
    match kind {
        0 => {
            let mut pad = [0u8; 3];
            rd.read_exact(&mut pad).await?;
            Ok(ClientMsg::SetPixelFormat(read_pixel_format(rd).await?))
        }
        2 => {
            let _pad = rd.read_u8().await?;
            let count = rd.read_u16().await?;
            let mut encodings = Vec::with_capacity(count as usize);
            for _ in 0..count {
                encodings.push(rd.read_i32().await?);
            }
            Ok(ClientMsg::SetEncodings(encodings))
        }
        3 => {
            let incremental = rd.read_u8().await? != 0;
            let _x = rd.read_u16().await?;
            let _y = rd.read_u16().await?;
            let _w = rd.read_u16().await?;
            let _h = rd.read_u16().await?;
            Ok(ClientMsg::FbUpdateRequest { incremental })
        }
        4 => {
            let down = rd.read_u8().await? != 0;
            let _pad = rd.read_u16().await?;
            let keysym = rd.read_u32().await?;
            Ok(ClientMsg::Key { down, keysym })
        }
        5 => {
            let mask = rd.read_u8().await?;
            let x = rd.read_u16().await?;
            let y = rd.read_u16().await?;
            Ok(ClientMsg::Pointer { mask, x, y })
        }
        6 => {
            let mut pad = [0u8; 3];
            rd.read_exact(&mut pad).await?;
            let len = rd.read_u32().await?;
            let mut buf = vec![0u8; len as usize];
            rd.read_exact(&mut buf).await?;
            Ok(ClientMsg::CutText)
        }
        other => Err(std::io::Error::new(
            std::io::ErrorKind::InvalidData,
            format!("unsupported VNC client message type {other}"),
        )),
    }
}

// --- Framebuffer updates ----------------------------------------------------

/// The displayed (upright) size of a native frame given the rotation.
fn display_dims(frame: &Frame, turns: u8) -> (u16, u16) {
    let (w, h) = (frame.width as u16, frame.height as u16);
    if turns % 2 == 1 { (h, w) } else { (w, h) }
}

/// Render a native frame into client pixel-format bytes, rotated upright and
/// clipped to `bound_w`×`bound_h`. Returns the rect's width, height, and bytes.
fn render(
    frame: &Frame,
    turns: u8,
    pf: &PixelFormat,
    bound_w: u16,
    bound_h: u16,
) -> (u16, u16, Vec<u8>) {
    let (dw, dh) = display_dims(frame, turns);
    let w = dw.min(bound_w);
    let h = dh.min(bound_h);
    let (nw, nh) = (frame.width, frame.height);

    if turns.is_multiple_of(4)
        && pf.bits_per_pixel == 32
        && !pf.big_endian
        && (pf.red_max, pf.green_max, pf.blue_max) == (255, 255, 255)
    {
        let (w, h) = (w as usize, h as usize);
        let (rs, gs, bs) = (pf.red_shift, pf.green_shift, pf.blue_shift);
        let mut out = vec![0u8; w * h * 4];
        for oy in 0..h {
            let src = &frame.rgba[oy * nw * 4..];
            let dst = &mut out[oy * w * 4..];
            for ox in 0..w {
                let s = &src[ox * 4..];
                let v = (s[0] as u32) << rs | (s[1] as u32) << gs | (s[2] as u32) << bs;
                dst[ox * 4..ox * 4 + 4].copy_from_slice(&v.to_le_bytes());
            }
        }
        return (w as u16, h as u16, out);
    }

    let mut out = Vec::with_capacity(w as usize * h as usize * pf.bytes_per_pixel());
    for oy in 0..h {
        for ox in 0..w {
            let (sx, sy) = if turns.is_multiple_of(4) {
                (ox as usize, oy as usize)
            } else {
                // Map displayed pixel back to native, using full (unclipped)
                // display dims so the aspect is right.
                let fx = (ox as f32 + 0.5) / dw as f32;
                let fy = (oy as f32 + 0.5) / dh as f32;
                let (nx, ny) = unrotate_norm(fx, fy, turns);
                (
                    ((nx * nw as f32) as usize).min(nw - 1),
                    ((ny * nh as f32) as usize).min(nh - 1),
                )
            };
            let idx = (sy * nw + sx) * 4;
            pf.put(
                &mut out,
                frame.rgba[idx],
                frame.rgba[idx + 1],
                frame.rgba[idx + 2],
            );
        }
    }
    (w, h, out)
}

/// Build a solid-colour `w`×`h` rectangle in the client pixel format.
fn solid_rect(pf: &PixelFormat, w: u16, h: u16, r: u8, g: u8, b: u8) -> Vec<u8> {
    let mut out = Vec::with_capacity(w as usize * h as usize * pf.bytes_per_pixel());
    for _ in 0..(w as usize * h as usize) {
        pf.put(&mut out, r, g, b);
    }
    out
}

/// One encoded rectangle ready for the wire: its position/size, RFB encoding
/// number, and the post-header payload (Raw pixels, or the Zlib length+stream
/// bytes already framed).
struct EncodedRect {
    x: u16,
    y: u16,
    w: u16,
    h: u16,
    encoding: i32,
    data: Vec<u8>,
}

/// Send one `FramebufferUpdate`: an optional desktop-size pseudo-rect followed by
/// `rects` pixel rectangles in whatever encoding each was built with.
async fn send_rects(
    wr: &mut OwnedWriteHalf,
    resize: Option<(u16, u16, bool)>,
    rects: &[EncodedRect],
) -> std::io::Result<()> {
    let count = resize.is_some() as u16 + rects.len() as u16;
    wr.write_u8(0).await?; // message-type: FramebufferUpdate
    wr.write_u8(0).await?; // padding
    wr.write_u16(count).await?;
    if let Some((w, h, ext)) = resize {
        write_resize_rect(wr, w, h, ext).await?;
    }
    for r in rects {
        wr.write_u16(r.x).await?;
        wr.write_u16(r.y).await?;
        wr.write_u16(r.w).await?;
        wr.write_u16(r.h).await?;
        wr.write_i32(r.encoding).await?;
        wr.write_all(&r.data).await?;
    }
    wr.flush().await
}

const TILE: usize = 64;

fn dirty_rects(
    prev: &[u8],
    cur: &[u8],
    w: usize,
    h: usize,
    bpp: usize,
) -> Vec<(usize, usize, usize, usize)> {
    let stride = w * bpp;
    let mut rects = Vec::new();
    let mut ty = 0;
    while ty < h {
        let th = TILE.min(h - ty);
        let mut run_start: Option<usize> = None;
        let mut tx = 0;
        while tx < w {
            let tw = TILE.min(w - tx);
            let changed = (ty..ty + th).any(|y| {
                let row = y * stride;
                prev[row + tx * bpp..row + (tx + tw) * bpp]
                    != cur[row + tx * bpp..row + (tx + tw) * bpp]
            });
            match (changed, run_start) {
                (true, None) => run_start = Some(tx),
                (false, Some(s)) => {
                    rects.push((s, ty, tx - s, th));
                    run_start = None;
                }
                _ => {}
            }
            tx += tw;
        }
        if let Some(s) = run_start {
            rects.push((s, ty, w - s, th));
        }
        ty += th;
    }
    rects
}

fn extract_rect(
    buf: &[u8],
    fb_w: usize,
    bpp: usize,
    x: usize,
    y: usize,
    w: usize,
    h: usize,
) -> Vec<u8> {
    let stride = fb_w * bpp;
    let mut out = Vec::with_capacity(w * h * bpp);
    for row in y..y + h {
        let base = row * stride + x * bpp;
        out.extend_from_slice(&buf[base..base + w * bpp]);
    }
    out
}

fn deflate(z: &mut Compress, input: &[u8]) -> Vec<u8> {
    let mut out = Vec::with_capacity(64 + input.len() / 2);
    let start_in = z.total_in();
    loop {
        if out.len() == out.capacity() {
            out.reserve(out.capacity().max(1024));
        }
        let consumed = (z.total_in() - start_in) as usize;
        z.compress_vec(&input[consumed..], &mut out, FlushCompress::Sync)
            .expect("zlib compress is infallible for in-memory buffers");
        // After a Sync flush, we're done once all input is consumed and the
        // output buffer still had spare room (so the flush emitted everything).
        if (z.total_in() - start_in) as usize == input.len() && out.len() < out.capacity() {
            break;
        }
    }
    out
}

fn build_rects(
    cur: &[u8],
    prev: Option<&Vec<u8>>,
    dims: (usize, usize, usize),
    force_full: bool,
    use_zlib: bool,
    zlib: &mut Compress,
) -> Vec<EncodedRect> {
    let (w, h, bpp) = dims;
    let regions = match prev {
        Some(p) if !force_full && p.len() == cur.len() => dirty_rects(p, cur, w, h, bpp),
        _ => vec![(0, 0, w, h)],
    };
    regions
        .into_iter()
        .map(|(rx, ry, rw, rh)| {
            let pixels = extract_rect(cur, w, bpp, rx, ry, rw, rh);
            let (encoding, data) = if use_zlib {
                let comp = deflate(zlib, &pixels);
                let mut data = Vec::with_capacity(4 + comp.len());
                data.extend_from_slice(&(comp.len() as u32).to_be_bytes());
                data.extend_from_slice(&comp);
                (6, data)
            } else {
                (0, pixels)
            };
            EncodedRect {
                x: rx as u16,
                y: ry as u16,
                w: rw as u16,
                h: rh as u16,
                encoding,
                data,
            }
        })
        .collect()
}

/// Write a desktop-size pseudo-encoding rectangle: `ExtendedDesktopSize` (-308)
/// if `ext`, else the simpler `DesktopSize` (-223).
async fn write_resize_rect(
    wr: &mut OwnedWriteHalf,
    w: u16,
    h: u16,
    ext: bool,
) -> std::io::Result<()> {
    if ext {
        wr.write_u16(0).await?; // x-position: reason (0 = server-initiated)
        wr.write_u16(0).await?; // y-position: status
        wr.write_u16(w).await?;
        wr.write_u16(h).await?;
        wr.write_i32(-308).await?;
        wr.write_u8(1).await?; // number of screens
        wr.write_all(&[0u8; 3]).await?; // padding
        wr.write_u32(0).await?; // screen id
        wr.write_u16(0).await?; // screen x
        wr.write_u16(0).await?; // screen y
        wr.write_u16(w).await?;
        wr.write_u16(h).await?;
        wr.write_u32(0).await?; // flags
    } else {
        wr.write_u16(0).await?; // x
        wr.write_u16(0).await?; // y
        wr.write_u16(w).await?;
        wr.write_u16(h).await?;
        wr.write_i32(-223).await?;
    }
    Ok(())
}

// --- Input mapping ----------------------------------------------------------

/// Map an X11 keysym to a printable character (ASCII + Latin-1).
fn keysym_to_char(keysym: u32) -> Option<char> {
    match keysym {
        0x20..=0x7e => Some(keysym as u8 as char),
        0xa0..=0xff => char::from_u32(keysym),
        _ => None,
    }
}

/// Map a non-text keysym (Enter, arrows, ...) to a HID Keyboard/Keypad usage.
fn special_keysym_usage(keysym: u32) -> Option<u64> {
    Some(match keysym {
        0xff0d | 0xff8d => 0x28, // Return / KP_Enter
        0xff1b => 0x29,          // Escape
        0xff08 => 0x2a,          // BackSpace
        0xff09 => 0x2b,          // Tab
        0xffff => 0x4c,          // Delete
        0xff53 => 0x4f,          // Right
        0xff51 => 0x50,          // Left
        0xff54 => 0x51,          // Down
        0xff52 => 0x52,          // Up
        0xff50 => 0x4a,          // Home
        0xff57 => 0x4d,          // End
        0xff55 => 0x4b,          // PageUp
        0xff56 => 0x4e,          // PageDown
        _ => return None,
    })
}

/// Map a keysym to a HID usage for a modifier chord (⌘ C, ⌘ Space...); also covers
/// letters/digits/space since the character is meaningless under a chord.
fn combo_keysym_usage(keysym: u32) -> Option<u64> {
    Some(match keysym {
        0x61..=0x7a => 0x04 + (keysym as u64 - 0x61), // a-z
        0x41..=0x5a => 0x04 + (keysym as u64 - 0x41), // A-Z
        0x31..=0x39 => 0x1e + (keysym as u64 - 0x31), // 1-9
        0x30 => 0x27,                                 // 0
        0x20 => 0x2c,                                 // space
        0x2d => 0x2d,                                 // minus
        0x3d => 0x2e,                                 // equals
        _ => return special_keysym_usage(keysym),
    })
}

/// Translate a key *press* into an `InputCmd`, given the modifiers held.
fn key_event_to_cmd(keysym: u32, mods: KeyMods) -> Option<InputCmd> {
    if mods.cmd || mods.ctrl || mods.alt {
        Some(InputCmd::KeyCombo {
            usage: combo_keysym_usage(keysym)?,
            mods,
        })
    } else if let Some(usage) = special_keysym_usage(keysym) {
        Some(InputCmd::KeyUsage(usage))
    } else {
        keysym_to_char(keysym).map(|c| InputCmd::Text(c.to_string()))
    }
}

// --- Connection -------------------------------------------------------------

async fn handle_client(
    stream: TcpStream,
    frames: FrameSlot,
    input: InputSink,
    orientation: OrientationSlot,
    password: String,
) -> std::io::Result<()> {
    let (mut rd, mut wr) = stream.into_split();

    // --- Handshake ---
    // Propose RFB 3.8 but adapt to the client's answer: security finalization
    // differs by version (see below) and getting it wrong desyncs the stream.
    wr.write_all(b"RFB 003.008\n").await?;
    wr.flush().await?;
    let mut ver = [0u8; 12];
    rd.read_exact(&mut ver).await?;
    tracing::info!(
        "VNC client version {}",
        String::from_utf8_lossy(&ver).trim_end()
    );
    let parse = |r: std::ops::Range<usize>| {
        ver.get(r)
            .and_then(|b| std::str::from_utf8(b).ok())
            .and_then(|s| s.parse::<u32>().ok())
    };
    let major = parse(4..7).unwrap_or(3);
    let minor = parse(8..11).unwrap_or(3);
    let is_38 = major > 3 || minor >= 8;
    let is_37plus = major > 3 || minor >= 7;

    // Security types (RFC 6143 §7.2).
    const SEC_NONE: u8 = 1;
    const SEC_VNC_AUTH: u8 = 2;

    // VNC auth is always offered (and is the only pre-3.7 choice) because many
    // older clients mishandle a None-only server and hang on "connecting".
    let require_auth = !password.is_empty();
    let chosen = if is_37plus {
        // Omit "None" when a password is required so it can't be picked to bypass auth.
        if require_auth {
            wr.write_all(&[1u8, SEC_VNC_AUTH]).await?;
        } else {
            wr.write_all(&[2u8, SEC_NONE, SEC_VNC_AUTH]).await?;
        }
        wr.flush().await?;
        rd.read_u8().await?
    } else {
        // 3.3: the server dictates a single type as a u32.
        wr.write_u32(SEC_VNC_AUTH as u32).await?;
        wr.flush().await?;
        SEC_VNC_AUTH
    };
    tracing::info!("VNC security type: {chosen}");

    if chosen == SEC_VNC_AUTH {
        let challenge = random_challenge();
        wr.write_all(&challenge).await?;
        wr.flush().await?;
        let mut response = [0u8; 16];
        rd.read_exact(&mut response).await?;

        if require_auth && response != vnc_auth_response(&password, &challenge) {
            // SecurityResult: failed (1). RFB 3.8 appends a reason string.
            wr.write_u32(1).await?;
            if is_38 {
                let reason = b"Authentication failed";
                wr.write_u32(reason.len() as u32).await?;
                wr.write_all(reason).await?;
            }
            wr.flush().await?;
            return Err(std::io::Error::new(
                std::io::ErrorKind::PermissionDenied,
                "VNC authentication failed",
            ));
        }
        wr.write_u32(0).await?; // SecurityResult: OK
        wr.flush().await?;
    } else if is_38 {
        // None: 3.8 sends a SecurityResult; 3.7 and 3.3 must not.
        wr.write_u32(0).await?; // OK
        wr.flush().await?;
    }

    // ClientInit (shared-flag) -> ServerInit.
    let _shared = rd.read_u8().await?;

    // Send ServerInit immediately - never block waiting for a frame, or clients
    // sit on "connecting". Size is corrected later via a desktop-size update.
    let mut pf = PixelFormat::server_default();
    let mut last_sent_version = 0u64;
    let (mut fb_w, mut fb_h) = match frames.latest() {
        Some((_, frame)) => display_dims(&frame, orientation.get().quarter_turns_cw()),
        None => DEFAULT_FB,
    };

    wr.write_u16(fb_w).await?;
    wr.write_u16(fb_h).await?;
    write_pixel_format(&mut wr, &pf).await?;
    let name = b"DeviceHub";
    wr.write_u32(name.len() as u32).await?;
    wr.write_all(name).await?;
    wr.flush().await?;
    tracing::info!("VNC handshake complete; serving {fb_w}x{fb_h}");

    // --- Session state ---
    let mut supports_ext_size = false;
    let mut supports_size = false;
    // Whether the client offered the Zlib encoding (RFB 6). One persistent zlib
    // stream serves the whole connection; its history is what lets near-identical
    // successive frames compress to almost nothing.
    let mut use_zlib = false;
    // Level 1 ("best speed"): it must keep up with the frame rate on this task, and
    // UI content still compresses heavily; higher levels cost CPU (latency) for
    // little gain on screen captures.
    let mut zlib = Compress::new(Compression::new(1), true);
    // The last framebuffer we sent, with its dimensions, in the client's pixel
    // format — kept to diff the next frame against. Reset to `None` (forcing a
    // full update) on a pixel-format change, and only used when the new frame's
    // dimensions match exactly, so we never diff across mismatched layouts.
    let mut last_sent: Option<(u16, u16, Vec<u8>)> = None;
    let mut pending = false; // a FramebufferUpdateRequest is outstanding
    let mut pending_full = false; // ...and it was non-incremental (send even if unchanged)
    // Until we've sent any framebuffer, answer a request with a blank frame so a
    // client blocked on its first update doesn't hang on "connecting".
    let mut sent_anything = false;
    // Pointer: a single synthetic finger driven by the left button.
    let mut touching = false;
    let mut last_touch = (0u16, 0u16);
    // Right button repurposed as the home button; edge-tracked so a held button
    // fires once, not on every pointer report.
    let mut right_down = false;
    let mut mods = KeyMods::default();

    let mut tick = tokio::time::interval(FRAME_INTERVAL);

    // Read client messages in their own task, delivering whole messages over a
    // channel. `read_client_message` is not cancel-safe — if `select!` dropped it
    // mid-message (a message split across TCP segments, parked on `read_exact`,
    // while the frame tick fires), the bytes already consumed would be lost and
    // every later message would be misparsed, corrupting `pf`/encodings and thus
    // the rendered image. Channel `recv` *is* cancel-safe, so the tick can never
    // truncate a read. The task is aborted when this handler returns.
    let (msg_tx, mut msg_rx) = tokio::sync::mpsc::unbounded_channel();
    let _reader = AbortOnDrop(tokio::spawn(async move {
        while let Ok(msg) = read_client_message(&mut rd).await {
            if msg_tx.send(msg).is_err() {
                break; // handler gone
            }
        }
        // On read error/EOF, dropping `msg_tx` closes the channel, which the
        // handler observes as `None` and treats as a disconnect.
    }));

    loop {
        tokio::select! {
            msg = msg_rx.recv() => {
                let Some(msg) = msg else {
                    return Ok(()); // client disconnected or sent a bad message
                };
                match msg {
                    ClientMsg::SetPixelFormat(p) => {
                        pf = p;
                        last_sent = None; // new layout: next update must be full
                    }
                    ClientMsg::SetEncodings(encs) => {
                        supports_ext_size = encs.contains(&-308);
                        supports_size = encs.contains(&-223);
                        use_zlib = ALLOW_ZLIB && encs.contains(&6);
                    }
                    ClientMsg::FbUpdateRequest { incremental } => {
                        pending = true;
                        if !incremental {
                            pending_full = true;
                        }
                    }
                    ClientMsg::Key { down, keysym } => match keysym {
                        0xffe1 | 0xffe2 => mods.shift = down,
                        0xffe3 | 0xffe4 => mods.ctrl = down,
                        0xffe9 | 0xffea | 0xff7e => mods.alt = down,
                        0xffe7 | 0xffe8 | 0xffeb | 0xffec => mods.cmd = down,
                        _ => {
                            if down && let Some(cmd) = key_event_to_cmd(keysym, mods) {
                                input.send(cmd);
                            }
                        }
                    },
                    ClientMsg::Pointer { mask, x, y } => {
                        let turns = orientation.get().quarter_turns_cw();
                        let fx = ((x as f32 + 0.5) / fb_w as f32).clamp(0.0, 1.0);
                        let fy = ((y as f32 + 0.5) / fb_h as f32).clamp(0.0, 1.0);
                        let (nx, ny) = unrotate_norm(fx, fy, turns);
                        let (px, py) = (norm(nx), norm(ny));
                        // Right button (bit 1) -> home, on the press edge only.
                        let right = mask & 2 != 0;
                        if right && !right_down {
                            input.send(InputCmd::Button("home"));
                        }
                        right_down = right;
                        let left = mask & 1 != 0;
                        match (touching, left) {
                            (false, true) => {
                                input.send(InputCmd::TouchDown { x: px, y: py });
                                touching = true;
                                last_touch = (px, py);
                            }
                            (true, true) => {
                                if (px, py) != last_touch {
                                    input.send(InputCmd::TouchMove { x: px, y: py });
                                    last_touch = (px, py);
                                }
                            }
                            (true, false) => {
                                input.send(InputCmd::TouchUp { x: px, y: py });
                                touching = false;
                            }
                            (false, false) => {}
                        }
                    }
                    ClientMsg::CutText => {}
                }
            }
            _ = tick.tick() => {
                if !pending {
                    continue;
                }
                match frames.latest() {
                    Some((version, frame)) => {
                        if !pending_full && sent_anything && version == last_sent_version {
                            continue; // nothing new since the last update
                        }
                        let turns = orientation.get().quarter_turns_cw();
                        let (dw, dh) = display_dims(&frame, turns);

                        // On a size change, renegotiate via desktop-size if the
                        // client supports it; otherwise keep the size and clip.
                        let mut resize = None;
                        let mut force_full = pending_full || !sent_anything;
                        if (dw, dh) != (fb_w, fb_h) && (supports_ext_size || supports_size) {
                            fb_w = dw;
                            fb_h = dh;
                            resize = Some((dw, dh, supports_ext_size));
                            force_full = true; // can't diff across a size change
                        }

                        let (w, h, buf) = render(&frame, turns, &pf, fb_w, fb_h);
                        let bpp = pf.bytes_per_pixel();
                        // Only diff against the previous frame when its dimensions
                        // match exactly; otherwise force a full update.
                        let prev = match &last_sent {
                            Some((pw, ph, pbuf)) if (*pw, *ph) == (w, h) => Some(pbuf),
                            _ => None,
                        };
                        let rects = build_rects(
                            &buf, prev, (w as usize, h as usize, bpp),
                            force_full, use_zlib, &mut zlib,
                        );

                        // Nothing visibly changed (the diff fell entirely outside
                        // the clipped region). Refresh our reference and keep the
                        // request outstanding so we answer when real change lands.
                        if rects.is_empty() {
                            last_sent = Some((w, h, buf));
                            last_sent_version = version;
                            continue;
                        }

                        send_rects(&mut wr, resize, &rects).await?;
                        if !sent_anything {
                            tracing::info!("VNC sent first framebuffer update {w}x{h}");
                        }
                        last_sent = Some((w, h, buf));
                        last_sent_version = version;
                        sent_anything = true;
                        pending = false;
                        pending_full = false;
                    }
                    // No decoded frame yet: send one blank frame so the client
                    // finishes connecting, then wait for real video.
                    None if !sent_anything => {
                        // One-time placeholder, always Raw; leave `last_sent` unset
                        // so the first real frame is sent as a clean full update.
                        let blank = EncodedRect {
                            x: 0,
                            y: 0,
                            w: fb_w,
                            h: fb_h,
                            encoding: 0,
                            data: solid_rect(&pf, fb_w, fb_h, 16, 16, 16),
                        };
                        send_rects(&mut wr, None, &[blank]).await?;
                        tracing::info!("VNC sent blank {fb_w}x{fb_h} (no video yet)");
                        sent_anything = true;
                        pending = false;
                        pending_full = false;
                    }
                    None => {} // connected, still waiting for the first frame
                }
            }
        }
    }
}
