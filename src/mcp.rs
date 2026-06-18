// An MCP server letting an agent drive the connected device: screenshot, then
// tap/swipe/type/press, and screenshot again.
//
// Coordinates: tools take pixel coordinates in the displayed (upright) image and
// are converted to the device's native normalized touch space via
// `unrotate_norm`/`norm`, so taps land correctly regardless of rotation.

use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use base64::Engine;
use rmcp::{
    ErrorData as McpError, ServerHandler,
    handler::server::{router::tool::ToolRouter, wrapper::Parameters},
    model::{CallToolResult, Content, Implementation, ServerCapabilities, ServerInfo},
    schemars, tool, tool_handler, tool_router,
    transport::streamable_http_server::{
        StreamableHttpServerConfig, StreamableHttpService, session::local::LocalSessionManager,
    },
};
use serde::Deserialize;
use serde_json::json;
use tokio::sync::mpsc::UnboundedSender;

use crate::protocol::{
    ActiveSlot, ControlCmd, DeviceListSlot, ErrorSlot, Frame, FrameSlot, InputCmd, InputSink,
    Orientation, OrientationSlot, RotateDir, StatusSlot, norm, unrotate_norm,
};

/// Default bind address. Loopback-only; override with `DEVICEHUB_MCP_ADDR`
/// (no auth, so keep it on a trusted network).
const DEFAULT_ADDR: &str = "127.0.0.1:8009";

/// Hold the finger across this many contact samples per tap; iOS's touch
/// recognizer sometimes drops a lone discrete contact.
const TAP_HOLD_SAMPLES: u32 = 3;

/// Delay between successive touch samples within a tap (~ one HID report tick).
const TAP_SAMPLE_MS: u64 = 25;

const SETTLE_MIN: Duration = Duration::from_millis(200);

const SETTLE_MAX: Duration = Duration::from_millis(2600);

const TAP_CHANGED_DIFF: f32 = 6.0;

const SETTLE_POLL: Duration = Duration::from_millis(110);

const SETTLE_DIFF: f32 = 2.5;

const SETTLE_STABLE_SAMPLES: u32 = 3;

const GRID_STEP: u32 = 100;

const GRID_LABEL_EVERY: u32 = 2;

/// Default cap on the screenshot's longer edge, in pixels. Phone screens are
/// ~2.5–3 MP natively, which costs an LLM thousands of image tokens per look;
/// downscaling the long edge to this keeps UI legible at a fraction of the
/// token cost. Coordinates stay in full screen space regardless (see
/// `screenshot`). Override per call via `ScreenshotParams::max_dim`.
const DEFAULT_MAX_DIM: u32 = 1024;

/// The MCP tool server. Cloned per client session; slots are cheap `Arc`
/// handles, so every connection drives the same live device session.
#[derive(Clone)]
pub struct DeviceHub {
    frames: FrameSlot,
    input: InputSink,
    orientation: OrientationSlot,
    devices: DeviceListSlot,
    active: ActiveSlot,
    error: ErrorSlot,
    status: StatusSlot,
    control: UnboundedSender<ControlCmd>,
    last_image: Arc<Mutex<Option<(u32, u32)>>>,
    last_tap: Arc<Mutex<Option<(f32, f32)>>>,
    tool_router: ToolRouter<DeviceHub>,
}

// --- Tool parameter types ----------------------------------------------------

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ScreenshotParams {
    /// Overlay a labeled coordinate grid (default true) to make it easy to read
    /// off tap/swipe pixel coordinates. Set false for a clean, unannotated image.
    pub grid: Option<bool>,
    /// Cap on the longer edge of the returned image, in pixels (default 1024).
    /// The image is downscaled for transfer to save tokens, but tap/swipe
    /// coordinates remain in full screen space — read them off the grid, which
    /// is labeled in screen coordinates. Raise this only when you need to read
    /// fine detail (e.g. small text); set 0 for native resolution.
    pub max_dim: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TapParams {
    /// X pixel coordinate in the screenshot's coordinate space.
    pub x: f32,
    /// Y pixel coordinate in the screenshot's coordinate space.
    pub y: f32,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SwipeParams {
    /// X of the start point, in screenshot pixels.
    pub x1: f32,
    /// Y of the start point, in screenshot pixels.
    pub y1: f32,
    /// X of the end point, in screenshot pixels.
    pub x2: f32,
    /// Y of the end point, in screenshot pixels.
    pub y2: f32,
    /// How long the drag should take, in milliseconds (default 300). Longer =
    /// slower; iOS reads velocity for flicks/scroll momentum.
    pub duration_ms: Option<u64>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct TextParams {
    /// The text to type (printable ASCII).
    pub text: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct KeyParams {
    /// Key name: enter, escape, backspace, tab, delete, up, down, left, right,
    /// home, end, pageup, pagedown.
    pub key: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ButtonParams {
    /// Hardware button: home, lock, volume-up, volume-down, mute, siri, action.
    pub button: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RotateParams {
    /// Direction to rotate 90 degrees: "left" (counter-clockwise) or "right".
    pub direction: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ConnectParams {
    /// The UDID of the device to connect to (from `list_devices`).
    pub udid: String,
}

// --- Geometry helpers --------------------------------------------------------

/// The displayed (upright) size of a native frame given its rotation, in pixels.
fn display_dims(frame: &Frame, turns: u8) -> (u32, u32) {
    let (w, h) = (frame.width as u32, frame.height as u32);
    if turns % 2 == 1 { (h, w) } else { (w, h) }
}

fn frame_signature(frame: &Frame) -> Vec<u8> {
    const N: usize = 24; // N×N samples
    let (w, h) = (frame.width, frame.height);
    if w == 0 || h == 0 {
        return Vec::new();
    }
    let mut sig = Vec::with_capacity(N * N);
    for j in 0..N {
        let y = ((j * h) / N + h / (2 * N)).min(h - 1);
        for i in 0..N {
            let x = ((i * w) / N + w / (2 * N)).min(w - 1);
            let p = (y * w + x) * 4;
            // Rec.601-ish luma, integer-weighted (R*2 + G*5 + B) / 8.
            let luma = (frame.rgba[p] as u16 * 2
                + frame.rgba[p + 1] as u16 * 5
                + frame.rgba[p + 2] as u16)
                / 8;
            sig.push(luma as u8);
        }
    }
    sig
}

fn signature_diff(a: &[u8], b: &[u8]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return f32::INFINITY;
    }
    let sum: u32 = a
        .iter()
        .zip(b)
        .map(|(x, y)| (*x as i32 - *y as i32).unsigned_abs())
        .sum();
    sum as f32 / a.len() as f32
}

/// Render a native frame into upright RGBA at its displayed dimensions, undoing
/// the device rotation so the agent sees the screen the right way up.
fn render_upright(frame: &Frame, turns: u8) -> (u32, u32, Vec<u8>) {
    let (dw, dh) = display_dims(frame, turns);
    if turns.is_multiple_of(4) {
        return (dw, dh, frame.rgba.clone());
    }
    let (nw, nh) = (frame.width, frame.height);
    let mut out = vec![0u8; dw as usize * dh as usize * 4];
    for oy in 0..dh {
        for ox in 0..dw {
            let fx = (ox as f32 + 0.5) / dw as f32;
            let fy = (oy as f32 + 0.5) / dh as f32;
            let (nx, ny) = unrotate_norm(fx, fy, turns);
            let sx = ((nx * nw as f32) as usize).min(nw - 1);
            let sy = ((ny * nh as f32) as usize).min(nh - 1);
            let sidx = (sy * nw + sx) * 4;
            let didx = (oy as usize * dw as usize + ox as usize) * 4;
            out[didx..didx + 4].copy_from_slice(&frame.rgba[sidx..sidx + 4]);
        }
    }
    (dw, dh, out)
}

/// Downscale an upright RGBA buffer so its longer edge is at most `max_dim`,
/// preserving aspect ratio. Returns the buffer unchanged when `max_dim` is 0 or
/// the image already fits. Bilinear (`Triangle`) is a good speed/quality balance
/// for shrinking UI screenshots. The grid is drawn *after* this so its lines and
/// glyphs stay crisp at the output size.
fn downscale(rgba: Vec<u8>, w: u32, h: u32, max_dim: u32) -> (u32, u32, Vec<u8>) {
    let long = w.max(h);
    // Bail (unchanged) when disabled, already small enough, or — defensively —
    // the buffer doesn't match the dimensions `from_raw` would reject.
    if max_dim == 0 || long <= max_dim || rgba.len() != (w as usize * h as usize * 4) {
        return (w, h, rgba);
    }
    let scale = max_dim as f32 / long as f32;
    let nw = ((w as f32 * scale).round() as u32).max(1);
    let nh = ((h as f32 * scale).round() as u32).max(1);
    let img = image::RgbaImage::from_raw(w, h, rgba).expect("buffer length checked above");
    let resized = image::imageops::resize(&img, nw, nh, image::imageops::FilterType::Triangle);
    (nw, nh, resized.into_raw())
}

// --- Coordinate grid overlay -------------------------------------------------
// Drawn onto the upright screenshot so a model can read tap/swipe coordinates
// directly off the image. Self-contained 3×5 bitmap font for the digit labels.

/// 3×5 bitmap glyphs for digits 0-9. Each row is 3 bits, MSB leftmost.
const DIGITS: [[u8; 5]; 10] = [
    [0b111, 0b101, 0b101, 0b101, 0b111], // 0
    [0b010, 0b110, 0b010, 0b010, 0b111], // 1
    [0b111, 0b001, 0b111, 0b100, 0b111], // 2
    [0b111, 0b001, 0b111, 0b001, 0b111], // 3
    [0b101, 0b101, 0b111, 0b001, 0b001], // 4
    [0b111, 0b100, 0b111, 0b001, 0b111], // 5
    [0b111, 0b100, 0b111, 0b101, 0b111], // 6
    [0b111, 0b001, 0b010, 0b100, 0b100], // 7
    [0b111, 0b101, 0b111, 0b101, 0b111], // 8
    [0b111, 0b101, 0b111, 0b001, 0b111], // 9
];

/// Alpha-blend an RGB colour onto an opaque RGBA pixel; out-of-bounds ignored.
fn blend_px(buf: &mut [u8], w: u32, h: u32, x: i32, y: i32, c: [u8; 3], a: f32) {
    if x < 0 || y < 0 || x >= w as i32 || y >= h as i32 {
        return;
    }
    let i = (y as usize * w as usize + x as usize) * 4;
    for k in 0..3 {
        buf[i + k] = (buf[i + k] as f32 * (1.0 - a) + c[k] as f32 * a).round() as u8;
    }
}

/// Fill an axis-aligned rectangle with an opaque colour (clipped to the image).
#[allow(clippy::too_many_arguments)]
fn fill_rect(buf: &mut [u8], w: u32, h: u32, x: i32, y: i32, rw: u32, rh: u32, c: [u8; 3]) {
    for dy in 0..rh as i32 {
        for dx in 0..rw as i32 {
            blend_px(buf, w, h, x + dx, y + dy, c, 1.0);
        }
    }
}

/// Draw `value` as digits at the top-left `(x0, y0)`, scaled by `scale`, with a
/// filled background box for legibility over any content.
fn draw_number(buf: &mut [u8], w: u32, h: u32, x0: i32, y0: i32, scale: u32, value: u32) {
    const FG: [u8; 3] = [255, 255, 0];
    const BG: [u8; 3] = [0, 0, 0];
    let s = value.to_string();
    let glyph_w = 3 * scale;
    let gap = scale;
    let pad = scale as i32;
    let total_w = s.len() as u32 * (glyph_w + gap);
    fill_rect(
        buf,
        w,
        h,
        x0 - pad,
        y0 - pad,
        total_w + 2 * pad as u32,
        5 * scale + 2 * pad as u32,
        BG,
    );
    let mut cx = x0;
    for ch in s.bytes() {
        let glyph = DIGITS[(ch - b'0') as usize];
        for (row, bits) in glyph.iter().enumerate() {
            for col in 0..3u32 {
                if bits & (1 << (2 - col)) != 0 {
                    fill_rect(
                        buf,
                        w,
                        h,
                        cx + (col * scale) as i32,
                        y0 + (row as u32 * scale) as i32,
                        scale,
                        scale,
                        FG,
                    );
                }
            }
        }
        cx += (glyph_w + gap) as i32;
    }
}

/// Pixel width of `value` rendered by [`draw_number`] at `scale`.
fn number_width(value: u32, scale: u32) -> u32 {
    value.to_string().len() as u32 * (4 * scale)
}

/// Overlay a labeled coordinate grid on a (possibly downscaled) upright RGBA
/// image. Lines and labels are placed in *screen* coordinate space
/// (`dev_w`×`dev_h`) — the space tap/swipe expect — and mapped onto the
/// `img_w`×`img_h` buffer, so the labels read true screen coordinates even when
/// the image has been shrunk for transfer. Magenta lines every [`GRID_STEP`]
/// screen px (brighter every 5th), majors labeled on both opposing edges so a
/// label is always near wherever you're aiming. When the image isn't downscaled
/// (`img == dev`) the mapping is identity.
fn draw_grid(buf: &mut [u8], img_w: u32, img_h: u32, dev_w: u32, dev_h: u32) {
    const LINE: [u8; 3] = [255, 0, 170];
    let scale = (img_w / 200).max(3);
    let glyph_h = 5 * scale;
    let margin = (scale * 2) as i32;
    let top_y = margin;
    let bot_y = img_h as i32 - glyph_h as i32 - margin;
    // screen-coordinate → image-pixel scale factors
    let fx = img_w as f32 / dev_w as f32;
    let fy = img_h as f32 / dev_h as f32;

    let mut x = GRID_STEP;
    while x < dev_w {
        let major = x.is_multiple_of(GRID_STEP * GRID_LABEL_EVERY);
        let a = if major { 0.7 } else { 0.3 };
        let px = (x as f32 * fx).round() as i32;
        for y in 0..img_h {
            blend_px(buf, img_w, img_h, px, y as i32, LINE, a);
            if major {
                blend_px(buf, img_w, img_h, px - 1, y as i32, LINE, a);
            }
        }
        if major {
            draw_number(buf, img_w, img_h, px + 3, top_y, scale, x);
            draw_number(buf, img_w, img_h, px + 3, bot_y, scale, x);
        }
        x += GRID_STEP;
    }

    let mut y = GRID_STEP;
    while y < dev_h {
        let major = y.is_multiple_of(GRID_STEP * GRID_LABEL_EVERY);
        let a = if major { 0.7 } else { 0.3 };
        let py = (y as f32 * fy).round() as i32;
        for x in 0..img_w {
            blend_px(buf, img_w, img_h, x as i32, py, LINE, a);
            if major {
                blend_px(buf, img_w, img_h, x as i32, py - 1, LINE, a);
            }
        }
        if major {
            let right_x = img_w as i32 - number_width(y, scale) as i32 - margin;
            draw_number(buf, img_w, img_h, margin, py + 3, scale, y);
            draw_number(buf, img_w, img_h, right_x, py + 3, scale, y);
        }
        y += GRID_STEP;
    }
}

fn draw_marker(buf: &mut [u8], img_w: u32, img_h: u32, cx: i32, cy: i32) {
    const C: [u8; 3] = [0, 255, 255];
    let r = (img_w.max(img_h) / 64).max(7) as i32;
    // Ring: step in fine angular increments so it stays connected at this radius.
    let steps = (r * 8).max(64);
    for s in 0..steps {
        let ang = s as f32 / steps as f32 * std::f32::consts::TAU;
        let (sin, cos) = ang.sin_cos();
        let x = cx + (r as f32 * cos).round() as i32;
        let y = cy + (r as f32 * sin).round() as i32;
        blend_px(buf, img_w, img_h, x, y, C, 1.0);
        blend_px(buf, img_w, img_h, x + 1, y, C, 0.5);
        blend_px(buf, img_w, img_h, x, y + 1, C, 0.5);
    }
    // Crosshair through the centre, with a small gap so the exact point is visible.
    for d in -r..=r {
        if d.abs() < r / 4 {
            continue;
        }
        blend_px(buf, img_w, img_h, cx + d, cy, C, 0.9);
        blend_px(buf, img_w, img_h, cx, cy + d, C, 0.9);
    }
}

/// Map a key name to a HID Keyboard/Keypad usage.
fn key_usage(name: &str) -> Option<u64> {
    Some(match name.to_ascii_lowercase().as_str() {
        "enter" | "return" => 0x28,
        "escape" | "esc" => 0x29,
        "backspace" => 0x2a,
        "tab" => 0x2b,
        "delete" | "del" => 0x4c,
        "right" => 0x4f,
        "left" => 0x50,
        "down" => 0x51,
        "up" => 0x52,
        "home" => 0x4a,
        "end" => 0x4d,
        "pageup" => 0x4b,
        "pagedown" => 0x4e,
        _ => return None,
    })
}

/// Normalize a button name to one of the `InputCmd::Button` static labels (see
/// `NAMED_BUTTONS` in `session.rs`).
fn button_label(name: &str) -> Option<&'static str> {
    Some(match name.to_ascii_lowercase().replace('_', "-").as_str() {
        "home" => "home",
        "lock" | "power" | "sleep" => "lock",
        "volume-up" | "volup" => "volume-up",
        "volume-down" | "voldown" => "volume-down",
        "mute" | "ring" => "mute",
        "siri" => "siri",
        "action" => "action",
        _ => return None,
    })
}

fn ok_text(s: impl Into<String>) -> Result<CallToolResult, McpError> {
    Ok(CallToolResult::success(vec![Content::text(s.into())]))
}

// --- Tools -------------------------------------------------------------------

#[tool_router]
impl DeviceHub {
    #[allow(clippy::too_many_arguments)]
    fn new(
        frames: FrameSlot,
        input: InputSink,
        orientation: OrientationSlot,
        devices: DeviceListSlot,
        active: ActiveSlot,
        error: ErrorSlot,
        status: StatusSlot,
        control: UnboundedSender<ControlCmd>,
    ) -> Self {
        Self {
            frames,
            input,
            orientation,
            devices,
            active,
            error,
            status,
            control,
            last_image: Arc::new(Mutex::new(None)),
            last_tap: Arc::new(Mutex::new(None)),
            tool_router: Self::tool_router(),
        }
    }

    /// Convert a displayed pixel coordinate to the device's native normalized
    /// touch coordinate. `None` when no frame has decoded yet.
    fn to_device(&self, x: f32, y: f32) -> Option<(u16, u16)> {
        let (_, frame) = self.frames.latest()?;
        let turns = self.orientation.get().quarter_turns_cw();
        let (dw, dh) = self
            .last_image
            .lock()
            .unwrap()
            .unwrap_or_else(|| display_dims(&frame, turns));
        let fx = ((x + 0.5) / dw as f32).clamp(0.0, 1.0);
        let fy = ((y + 0.5) / dh as f32).clamp(0.0, 1.0);
        let (nx, ny) = unrotate_norm(fx, fy, turns);
        Some((norm(nx), norm(ny)))
    }

    async fn settle(&self) {
        tokio::time::sleep(SETTLE_MIN).await;
        let start = Instant::now();
        let mut prev = self.frames.latest().map(|(_, f)| frame_signature(&f));
        let mut stable = 0u32;
        while start.elapsed() < SETTLE_MAX {
            tokio::time::sleep(SETTLE_POLL).await;
            let cur = self.frames.latest().map(|(_, f)| frame_signature(&f));
            match (&prev, &cur) {
                (Some(a), Some(b)) if signature_diff(a, b) < SETTLE_DIFF => {
                    stable += 1;
                    if stable >= SETTLE_STABLE_SAMPLES {
                        break;
                    }
                }
                // Any motion (or a dropped frame) resets the quiet streak.
                _ => stable = 0,
            }
            prev = cur;
        }
    }

    #[tool(
        description = "Capture the current screen of the connected iPhone as a PNG. \
        The image is downscaled for transfer (longer edge capped at 1024px by \
        default; raise via `max_dim` for fine detail). Tap/swipe coordinates are \
        pixels in the returned image itself (origin top-left) — the size is \
        reported in the response text and the image is what you measure against, \
        so what you see is the coordinate space. By default a labeled coordinate \
        grid is overlaid so you can read tap/swipe coordinates directly off the \
        image. Call this before and after acting. Note that the device hides the \
        passcode input, so you will have to enter passcodes via the keyboard."
    )]
    async fn screenshot(
        &self,
        Parameters(ScreenshotParams { grid, max_dim }): Parameters<ScreenshotParams>,
    ) -> Result<CallToolResult, McpError> {
        let Some((_, frame)) = self.frames.latest() else {
            return ok_text(
                "No frame available yet — connect a device with `connect_device` and \
                 wait for the screen to start streaming.",
            );
        };
        let turns = self.orientation.get().quarter_turns_cw();
        // `(w, h)` is the full screen coordinate space; tap/swipe coordinates
        // live here regardless of how the transferred image is scaled.
        let (w, h, rgba) = render_upright(&frame, turns);

        // Downscale for transfer to save image tokens, then draw the grid in
        // screen coordinates onto the smaller buffer (so labels stay true and
        // glyphs stay crisp).
        let (iw, ih, mut rgba) = downscale(rgba, w, h, max_dim.unwrap_or(DEFAULT_MAX_DIM));
        // Tap/swipe coordinates live in the space of *this* image, so record its
        // dimensions for `to_device` and label the grid in image pixels.
        *self.last_image.lock().unwrap() = Some((iw, ih));
        let gridded = grid.unwrap_or(true);
        if gridded {
            draw_grid(&mut rgba, iw, ih, iw, ih);
        }
        let marked = if let Some((mfx, mfy)) = *self.last_tap.lock().unwrap() {
            let mx = (mfx * iw as f32).round() as i32;
            let my = (mfy * ih as f32).round() as i32;
            draw_marker(&mut rgba, iw, ih, mx, my);
            true
        } else {
            false
        };

        let mut png = Vec::new();
        use image::{ExtendedColorType, ImageEncoder, codecs::png::PngEncoder};
        PngEncoder::new(&mut png)
            .write_image(&rgba, iw, ih, ExtendedColorType::Rgba8)
            .map_err(|e| McpError::internal_error(format!("PNG encode failed: {e}"), None))?;
        let b64 = base64::engine::general_purpose::STANDARD.encode(&png);

        let mut note = if gridded {
            format!(
                "Image is {iw}x{ih} pixels (width x height); tap/swipe coordinates \
                 are pixels in THIS image (origin top-left). A coordinate grid is \
                 overlaid: magenta lines every {GRID_STEP}px, with brighter lines \
                 labeled (yellow) every {}px — x values along the top and bottom \
                 edges, y values down the left and right edges. Read tap/swipe \
                 coordinates directly off the grid.",
                GRID_STEP * GRID_LABEL_EVERY
            )
        } else {
            format!(
                "Image is {iw}x{ih} pixels (width x height); tap/swipe coordinates \
                 are pixels in THIS image (origin top-left)."
            )
        };
        if marked {
            note.push_str(
                " The cyan ◯ crosshair marks where your last tap landed. The screen \
                 didn't change after that tap, so it likely missed — check the \
                 marker against your intended target and re-tap with a corrected \
                 coordinate.",
            );
        }
        Ok(CallToolResult::success(vec![
            Content::text(note),
            Content::image(b64, "image/png".to_string()),
        ]))
    }

    /// Touch down, hold across a few contact samples so iOS reliably registers
    /// the touch, then lift.
    async fn press(&self, x: u16, y: u16) {
        self.input.send(InputCmd::TouchDown { x, y });
        for _ in 0..TAP_HOLD_SAMPLES {
            tokio::time::sleep(Duration::from_millis(TAP_SAMPLE_MS)).await;
            self.input.send(InputCmd::TouchMove { x, y });
        }
        tokio::time::sleep(Duration::from_millis(TAP_SAMPLE_MS)).await;
        self.input.send(InputCmd::TouchUp { x, y });
    }

    #[tool(description = "Tap the screen once at a pixel coordinate from the screenshot.")]
    async fn tap(
        &self,
        Parameters(TapParams { x, y }): Parameters<TapParams>,
    ) -> Result<CallToolResult, McpError> {
        let Some((px, py)) = self.to_device(x, y) else {
            return ok_text("No screen available — connect a device first.");
        };
        if let Some((iw, ih)) = *self.last_image.lock().unwrap() {
            *self.last_tap.lock().unwrap() = Some((
                (x / iw as f32).clamp(0.0, 1.0),
                (y / ih as f32).clamp(0.0, 1.0),
            ));
        }
        let before = self.frames.latest().map(|(_, f)| frame_signature(&f));
        self.press(px, py).await;
        self.settle().await;
        let after = self.frames.latest().map(|(_, f)| frame_signature(&f));
        if let (Some(a), Some(b)) = (&before, &after)
            && signature_diff(a, b) >= TAP_CHANGED_DIFF
        {
            *self.last_tap.lock().unwrap() = None;
        }
        ok_text(format!("Tapped ({x}, {y})."))
    }

    #[tool(
        description = "Swipe/drag from one point to another (pixels). Use for scrolling, \
        swiping between pages, or pull-to-refresh. duration_ms controls speed."
    )]
    async fn swipe(
        &self,
        Parameters(SwipeParams {
            x1,
            y1,
            x2,
            y2,
            duration_ms,
        }): Parameters<SwipeParams>,
    ) -> Result<CallToolResult, McpError> {
        let Some((sx, sy)) = self.to_device(x1, y1) else {
            return ok_text("No screen available — connect a device first.");
        };
        // A swipe isn't a single point; drop any stale tap marker.
        *self.last_tap.lock().unwrap() = None;
        let dur = duration_ms.unwrap_or(300).clamp(50, 5000);
        // ~60 Hz of move samples; iOS reads the velocity for momentum.
        let steps = (dur / 16).clamp(2, 150);
        self.input.send(InputCmd::TouchDown { x: sx, y: sy });
        for i in 1..=steps {
            let t = i as f32 / steps as f32;
            let xi = x1 + (x2 - x1) * t;
            let yi = y1 + (y2 - y1) * t;
            if let Some((mx, my)) = self.to_device(xi, yi) {
                self.input.send(InputCmd::TouchMove { x: mx, y: my });
            }
            tokio::time::sleep(Duration::from_millis(dur / steps)).await;
        }
        if let Some((ex, ey)) = self.to_device(x2, y2) {
            self.input.send(InputCmd::TouchUp { x: ex, y: ey });
        }
        self.settle().await;
        ok_text(format!("Swiped ({x1}, {y1}) → ({x2}, {y2}) over {dur}ms."))
    }

    #[tool(description = "Type printable text into the currently focused field.")]
    async fn type_text(
        &self,
        Parameters(TextParams { text }): Parameters<TextParams>,
    ) -> Result<CallToolResult, McpError> {
        let n = text.chars().count();
        self.input.send(InputCmd::Text(text));
        ok_text(format!("Typed {n} characters."))
    }

    #[tool(
        description = "Press a special key: enter, escape, backspace, tab, delete, \
        up, down, left, right, home, end, pageup, pagedown."
    )]
    async fn press_key(
        &self,
        Parameters(KeyParams { key }): Parameters<KeyParams>,
    ) -> Result<CallToolResult, McpError> {
        let Some(usage) = key_usage(&key) else {
            return ok_text(format!("Unknown key '{key}'."));
        };
        self.input.send(InputCmd::KeyUsage(usage));
        self.settle().await;
        ok_text(format!("Pressed {key}."))
    }

    #[tool(
        description = "Press a hardware button: home, lock, volume-up, volume-down, \
        mute, siri, action. 'home' returns to the home screen."
    )]
    async fn press_button(
        &self,
        Parameters(ButtonParams { button }): Parameters<ButtonParams>,
    ) -> Result<CallToolResult, McpError> {
        let Some(label) = button_label(&button) else {
            return ok_text(format!("Unknown button '{button}'."));
        };
        *self.last_tap.lock().unwrap() = None;
        self.input.send(InputCmd::Button(label));
        self.settle().await;
        ok_text(format!("Pressed {label} button."))
    }

    #[tool(description = "Rotate the device 90°: direction 'left' or 'right'.")]
    async fn rotate(
        &self,
        Parameters(RotateParams { direction }): Parameters<RotateParams>,
    ) -> Result<CallToolResult, McpError> {
        let dir = match direction.to_ascii_lowercase().as_str() {
            "left" | "ccw" | "counterclockwise" => RotateDir::Left,
            "right" | "cw" | "clockwise" => RotateDir::Right,
            _ => return ok_text(format!("Unknown direction '{direction}' (use left/right).")),
        };
        // Orientation change invalidates the upright-space tap fraction.
        *self.last_tap.lock().unwrap() = None;
        self.input.send(InputCmd::Rotate(dir));
        self.settle().await;
        ok_text(format!("Rotated {direction}."))
    }

    #[tool(description = "List the iOS devices currently attached (UDID, name, USB/Wi-Fi).")]
    async fn list_devices(&self) -> Result<CallToolResult, McpError> {
        let active = self.active.get();
        let devices: Vec<_> = self
            .devices
            .get()
            .into_iter()
            .map(|d| {
                json!({
                    "udid": d.udid,
                    "name": d.name,
                    "connection": d.connection.label(),
                    "active": active.as_deref() == Some(d.udid.as_str()),
                })
            })
            .collect();
        ok_text(json!({ "devices": devices }).to_string())
    }

    #[tool(
        description = "Connect to a device by UDID and wait for its screen to start \
        streaming. Returns once the live screen is available (or times out)."
    )]
    async fn connect_device(
        &self,
        Parameters(ConnectParams { udid }): Parameters<ConnectParams>,
    ) -> Result<CallToolResult, McpError> {
        if self
            .control
            .send(ControlCmd::Connect(udid.clone()))
            .is_err()
        {
            return Err(McpError::internal_error(
                "device session manager is not running",
                None,
            ));
        }
        for _ in 0..40 {
            if let Some(err) = self.error.get() {
                return ok_text(format!("Failed to connect to {udid}: {err}"));
            }
            if self.active.get().as_deref() == Some(udid.as_str()) && self.frames.latest().is_some()
            {
                return ok_text(format!("Connected to {udid}; screen is streaming."));
            }
            tokio::time::sleep(Duration::from_millis(500)).await;
        }
        ok_text(format!(
            "Requested connection to {udid}; still establishing the stream — call \
             `screenshot` shortly to check."
        ))
    }

    #[tool(
        description = "Report connection status: which device is active, the stream \
        status text, the current screen size, and orientation."
    )]
    async fn status(&self) -> Result<CallToolResult, McpError> {
        let (has_frame, dims) = match self.frames.latest() {
            Some((_, frame)) => {
                let (w, h) = display_dims(&frame, self.orientation.get().quarter_turns_cw());
                (true, json!([w, h]))
            }
            None => (false, json!(null)),
        };
        let orientation = match self.orientation.get() {
            Orientation::Portrait => "portrait",
            Orientation::PortraitUpsideDown => "portrait-upside-down",
            Orientation::LandscapeLeft => "landscape-left",
            Orientation::LandscapeRight => "landscape-right",
        };
        ok_text(
            json!({
                "active_udid": self.active.get(),
                "status": self.status.get(),
                "error": self.error.get(),
                "streaming": has_frame,
                "screen_size": dims,
                "orientation": orientation,
            })
            .to_string(),
        )
    }
}

// Reuse the router built once at construction; the macro default would rebuild
// it on every call.
#[tool_handler(router = self.tool_router)]
impl ServerHandler for DeviceHub {
    fn get_info(&self) -> ServerInfo {
        ServerInfo::new(ServerCapabilities::builder().enable_tools().build())
            .with_server_info(Implementation::from_build_env())
            .with_instructions(
                "Control a connected iPhone like a person: call `screenshot` to see the \
                 screen, then `tap`/`swipe`/`type_text`/`press_key`/`press_button` to act, \
                 and `screenshot` again to observe the result. Tap/swipe coordinates are \
                 pixels in the most recent screenshot. Small targets like app icons are \
                 easy to misjudge — after a `tap`, the next `screenshot` shows a cyan ◯ \
                 marker where your tap actually landed; if it's off the target, re-tap with \
                 a corrected coordinate before moving on. Actions wait for on-screen \
                 animations to settle before returning, so the next screenshot is current. \
                 Use `list_devices` and `connect_device` to choose a device if none is \
                 connected."
                    .to_string(),
            )
    }
}

/// Serve the MCP tool server over streamable HTTP for the app's lifetime. Binds
/// `DEFAULT_ADDR` (override via `DEVICEHUB_MCP_ADDR`); a bind failure is logged
/// and the server doesn't come up.
#[allow(clippy::too_many_arguments)]
pub async fn serve(
    frames: FrameSlot,
    input: InputSink,
    orientation: OrientationSlot,
    devices: DeviceListSlot,
    active: ActiveSlot,
    error: ErrorSlot,
    status: StatusSlot,
    control: UnboundedSender<ControlCmd>,
) {
    let addr = std::env::var("DEVICEHUB_MCP_ADDR").unwrap_or_else(|_| DEFAULT_ADDR.to_string());

    let hub = DeviceHub::new(
        frames,
        input,
        orientation,
        devices,
        active,
        error,
        status,
        control,
    );
    let service = StreamableHttpService::new(
        move || Ok(hub.clone()),
        LocalSessionManager::default().into(),
        StreamableHttpServerConfig::default(),
    );
    let router = axum::Router::new().nest_service("/mcp", service);

    match tokio::net::TcpListener::bind(&addr).await {
        Ok(listener) => {
            tracing::info!("MCP server listening on http://{addr}/mcp");
            if let Err(e) = axum::serve(listener, router).await {
                tracing::error!("MCP server stopped: {e}");
            }
        }
        Err(e) => tracing::warn!("MCP: failed to bind {addr}: {e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn all_tools_registered() {
        let names: Vec<String> = DeviceHub::tool_router()
            .list_all()
            .into_iter()
            .map(|t| t.name.to_string())
            .collect();
        for expected in [
            "screenshot",
            "tap",
            "swipe",
            "type_text",
            "press_key",
            "press_button",
            "rotate",
            "list_devices",
            "connect_device",
            "status",
        ] {
            assert!(
                names.contains(&expected.to_string()),
                "missing tool: {expected}"
            );
        }
    }

    #[test]
    fn grid_overlay_stays_in_bounds() {
        // A frame size that isn't a multiple of GRID_STEP, to exercise the edges.
        let (w, h) = (393u32, 851u32);
        let mut rgba = vec![128u8; (w * h * 4) as usize];
        draw_grid(&mut rgba, w, h, w, h);
        assert_eq!(rgba.len(), (w * h * 4) as usize);
    }

    #[test]
    fn marker_stays_in_bounds_even_at_edges() {
        let (w, h) = (393u32, 851u32);
        let mut rgba = vec![128u8; (w * h * 4) as usize];
        for (cx, cy) in [(w as i32 / 2, h as i32 / 2), (0, 0), (-50, h as i32 + 80)] {
            draw_marker(&mut rgba, w, h, cx, cy);
        }
        assert_eq!(rgba.len(), (w * h * 4) as usize);
    }

    #[test]
    fn signature_detects_motion_and_ignores_noise() {
        let frame = |fill: u8| Frame {
            width: 64,
            height: 64,
            rgba: vec![fill; 64 * 64 * 4],
        };
        let base = frame_signature(&frame(100));
        // Identical frames: zero difference (settled).
        assert!(signature_diff(&base, &frame_signature(&frame(100))) < SETTLE_DIFF);
        // A 1-luma codec-noise wobble still counts as settled.
        assert!(signature_diff(&base, &frame_signature(&frame(101))) < SETTLE_DIFF);
        // A large change (animation) is well above threshold.
        assert!(signature_diff(&base, &frame_signature(&frame(180))) >= SETTLE_DIFF);
        // Mismatched/empty signatures are treated as "still moving".
        assert_eq!(signature_diff(&base, &[]), f32::INFINITY);
    }

    #[test]
    fn downscale_caps_long_edge_and_preserves_aspect() {
        let (w, h) = (1170u32, 2532u32);
        let rgba = vec![64u8; (w * h * 4) as usize];
        let (iw, ih, out) = downscale(rgba, w, h, 1024);
        assert_eq!(ih, 1024, "long edge should hit the cap");
        assert!(iw < w && ih < h, "both dimensions should shrink");
        // aspect ratio preserved within rounding
        let ar_in = w as f32 / h as f32;
        let ar_out = iw as f32 / ih as f32;
        assert!((ar_in - ar_out).abs() < 0.01);
        assert_eq!(out.len(), (iw * ih * 4) as usize);
    }

    #[test]
    fn downscale_noop_when_within_cap() {
        let (w, h) = (400u32, 800u32);
        let rgba = vec![64u8; (w * h * 4) as usize];
        let (iw, ih, out) = downscale(rgba, w, h, 1024);
        assert_eq!((iw, ih), (w, h));
        assert_eq!(out.len(), (w * h * 4) as usize);
    }

    #[test]
    fn grid_on_downscaled_buffer_stays_in_bounds() {
        // Screen space larger than the image buffer: grid drawn in screen
        // coords must map into the smaller buffer without overflowing it.
        let (dev_w, dev_h) = (1170u32, 2532u32);
        let (img_w, img_h) = (473u32, 1024u32);
        let mut rgba = vec![128u8; (img_w * img_h * 4) as usize];
        draw_grid(&mut rgba, img_w, img_h, dev_w, dev_h);
        assert_eq!(rgba.len(), (img_w * img_h * 4) as usize);
    }

    #[test]
    fn key_and_button_names_map() {
        assert_eq!(key_usage("enter"), Some(0x28));
        assert_eq!(key_usage("PageDown"), Some(0x4e));
        assert_eq!(key_usage("nope"), None);
        assert_eq!(button_label("volume_up"), Some("volume-up"));
        assert_eq!(button_label("power"), Some("lock"));
        assert_eq!(button_label("nope"), None);
    }
}
