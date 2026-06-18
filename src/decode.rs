// HEVC decode via an `ffmpeg` subprocess: Annex-B in on stdin, PAM (P7) frames
// out on stdout. PAM is self-describing (size in every header, so resolution
// changes need no extra signalling) and, unlike PPM, carries an alpha channel —
// so we ask ffmpeg for `rgba` directly and publish its bytes verbatim, with no
// per-pixel RGB→RGBA expansion pass on our side.

use std::sync::Arc;

use tokio::io::{AsyncBufReadExt, AsyncReadExt, BufReader};
use tokio::process::{Child, ChildStderr, ChildStdin, ChildStdout, Command};
use tokio::sync::Notify;

use crate::protocol::{Frame, FrameSlot};

/// Spawn `ffmpeg` decoding raw HEVC (Annex-B on stdin) to PAM frames on stdout.
/// stderr is piped so the session can watch it for decode errors.
pub fn spawn_ffmpeg() -> std::io::Result<(Child, ChildStdin, ChildStdout, ChildStderr)> {
    let mut child = Command::new("ffmpeg")
        // Do *not* add `-fflags nobuffer`: it makes ffmpeg skip the opening IDR +
        // parameter sets, so every P-frame fails with "Could not find ref".
        .args(["-flags", "low_delay"])
        .args(["-hwaccel", "auto"])
        .args(["-f", "hevc", "-i", "pipe:0"])
        .args([
            "-an",
            "-f",
            "image2pipe",
            "-vcodec",
            "pam",
            "-pix_fmt",
            "rgba",
        ])
        .arg("pipe:1")
        .args(["-loglevel", "error"])
        .stdin(std::process::Stdio::piped())
        .stdout(std::process::Stdio::piped())
        .stderr(std::process::Stdio::piped())
        .kill_on_drop(true)
        .spawn()?;

    let stdin = child.stdin.take().expect("ffmpeg stdin piped");
    let stdout = child.stdout.take().expect("ffmpeg stdout piped");
    let stderr = child.stderr.take().expect("ffmpeg stderr piped");
    Ok((child, stdin, stdout, stderr))
}

/// Read PAM frames from ffmpeg's stdout, publishing each (already RGBA) into
/// `slot` and waking the UI via `repaint`. Each frame pulses `beat`, the liveness
/// heartbeat watched by the session's stall watchdog. Returns when the stream
/// ends.
pub async fn read_frames(
    stdout: ChildStdout,
    slot: FrameSlot,
    beat: Arc<Notify>,
    repaint: impl Fn(),
) {
    let mut reader = BufReader::new(stdout);
    let mut last_dims: Option<(usize, usize)> = None;
    let mut read_buf: Vec<u8> = Vec::new();
    let mut last: Option<Arc<Frame>> = None;
    let mut pool: Vec<Vec<u8>> = Vec::new();
    loop {
        match read_pam(&mut reader, &mut read_buf).await {
            Ok(Some((width, height))) => {
                let dims = (width, height);
                if last_dims != Some(dims) {
                    tracing::info!("decoded frame size: {}x{}", dims.0, dims.1);
                    last_dims = Some(dims);
                }

                // Pulse even for duplicate frames: a frozen-but-streaming
                // screen is still a healthy stream.
                beat.notify_one();

                if last.as_ref().is_some_and(|p| p.rgba == read_buf) {
                    continue;
                }

                let frame = Arc::new(Frame {
                    width,
                    height,
                    rgba: std::mem::take(&mut read_buf),
                });
                read_buf = pool.pop().unwrap_or_default();
                last = Some(frame.clone());
                if let Some(prev) = slot.publish(frame)
                    && let Ok(frame) = Arc::try_unwrap(prev)
                    && pool.len() < 2
                {
                    pool.push(frame.rgba);
                }
                repaint();
            }
            Ok(None) => {
                tracing::info!("ffmpeg stdout closed");
                break;
            }
            Err(e) => {
                tracing::warn!("pam read error: {e}");
                break;
            }
        }
    }
}

/// Read a single binary PAM (P7) image into `rgba` as a raw top-down RGBA raster,
/// reusing its allocation. Returns the dimensions, or `Ok(None)` at clean EOF.
///
/// PAM headers are line-oriented: `P7`, then `KEY VALUE` lines (`WIDTH`,
/// `HEIGHT`, `DEPTH`, `MAXVAL`, `TUPLTYPE`) in any order, terminated by `ENDHDR`,
/// then the raster. We require the 4-channel/8-bit layout ffmpeg emits for `rgba`.
async fn read_pam<R: AsyncReadExt + AsyncBufReadExt + Unpin>(
    r: &mut R,
    rgba: &mut Vec<u8>,
) -> std::io::Result<Option<(usize, usize)>> {
    let invalid = |msg: String| std::io::Error::new(std::io::ErrorKind::InvalidData, msg);

    let mut line = Vec::new();
    // First line: the `P7` magic. Zero bytes here is a clean end-of-stream.
    if r.read_until(b'\n', &mut line).await? == 0 {
        return Ok(None);
    }
    if trim(&line) != b"P7" {
        return Err(invalid(format!(
            "expected PAM 'P7' magic, got {:?}",
            trim(&line)
        )));
    }

    let (mut width, mut height, mut depth, mut maxval) = (0usize, 0usize, 0usize, 0usize);
    loop {
        line.clear();
        if r.read_until(b'\n', &mut line).await? == 0 {
            return Err(std::io::Error::from(std::io::ErrorKind::UnexpectedEof));
        }
        let l = trim(&line);
        if l == b"ENDHDR" {
            break;
        }
        if l.is_empty() || l[0] == b'#' {
            continue;
        }
        // Parse `KEY VALUE`; only the numeric fields matter (TUPLTYPE is ignored).
        let text = String::from_utf8_lossy(l);
        let mut it = text.split_ascii_whitespace();
        let key = it.next().unwrap_or("");
        let slot = match key {
            "WIDTH" => &mut width,
            "HEIGHT" => &mut height,
            "DEPTH" => &mut depth,
            "MAXVAL" => &mut maxval,
            _ => continue,
        };
        *slot = it
            .next()
            .and_then(|v| v.parse().ok())
            .ok_or_else(|| invalid(format!("bad PAM header line: {text:?}")))?;
    }

    if depth != 4 || maxval != 255 {
        return Err(invalid(format!(
            "expected 8-bit 4-channel PAM, got depth={depth} maxval={maxval}"
        )));
    }
    if width == 0 || height == 0 {
        return Err(invalid(format!("bad PAM dimensions {width}x{height}")));
    }

    rgba.resize(width * height * depth, 0);
    r.read_exact(rgba).await?;
    Ok(Some((width, height)))
}

/// Strip a trailing `\n`/`\r\n` and surrounding ASCII whitespace from a header line.
fn trim(line: &[u8]) -> &[u8] {
    let mut s = line;
    while let [rest @ .., last] = s
        && last.is_ascii_whitespace()
    {
        s = rest;
    }
    while let [first, rest @ ..] = s
        && first.is_ascii_whitespace()
    {
        s = rest;
    }
    s
}
