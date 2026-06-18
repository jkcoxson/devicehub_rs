// The egui UI: render the live screen and translate host input into device HID
// commands.

use eframe::egui;
use tokio::sync::mpsc::UnboundedSender;

use crate::protocol::{
    ActiveSlot, ClipboardEvent, ClipboardSlot, ControlCmd, DeviceListSlot, ErrorSlot, FrameSlot,
    InputCmd, InputSink, KeyMods, OrientationSlot, RotateDir, StatusSlot, VncControl, norm,
    unrotate_norm,
};
use crate::session::NAMED_BUTTONS;

/// Seconds the clipboard-activity indicator stays fully visible before fading.
const CLIPBOARD_INDICATOR_TTL: f64 = 4.0;

/// On-screen finger movement per point of host scroll delta (host already bakes
/// in its own scroll acceleration).
const SCROLL_GAIN: f32 = 1.0;
/// Flip to `-1.0` if scrolling feels backwards on the device.
const SCROLL_INVERT: f32 = 1.0;
/// Fallback idle-lift timeout (seconds); a gesture normally ends on release.
const SCROLL_END_IDLE: f64 = 0.12;
/// Travel (after gain) required before planting the finger. Below this a flick
/// would land and lift near the same spot, which iOS reads as a tap.
const SCROLL_DEADZONE: f32 = 8.0;

// egui-winit (0.29) drops macOS's scroll phase, so we can't tell finger-on-
// trackpad from OS momentum replay. We infer release from velocity (in
// screens/sec, window-size independent): momentum only decelerates, so a fast
// gesture that starts slowing means release — lift while still moving to hand
// iOS a release velocity, then ignore the decaying tail.
/// Peak speed (screens/sec) above which a gesture is a flick eligible for
/// release-while-moving.
const FLICK_SPEED: f32 = 2.5;
/// Once a flick's speed drops below `peak * this`, treat it as released and lift.
const FLICK_RELEASE_RATIO: f32 = 0.8;
/// Speed (screens/sec) below which the finger is considered stopped.
const STOP_SPEED: f32 = 0.4;
/// While coasting, a sample faster than the last tail sample by this factor is
/// fresh input (momentum only decelerates) and cancels the coast.
const REENGAGE_RATIO: f32 = 1.25;

/// A two-finger-scroll gesture in progress. Travel accumulates in `pending` and
/// `TouchDown` only fires once it clears [`SCROLL_DEADZONE`], so a stray
/// micro-scroll never lands as a tap.
struct ScrollGesture {
    /// Where the finger first contacts (the cursor at gesture start).
    start: egui::Pos2,
    /// The synthetic finger's current position (valid once `down`).
    pos: egui::Pos2,
    /// Host time of the last scroll event (drives the idle-lift timer).
    last_event: f64,
    /// Travel accumulated while still inside the deadzone (before `down`).
    pending: egui::Vec2,
    /// Whether we've planted the finger (`TouchDown` sent) yet.
    down: bool,
    /// Smoothed finger speed and the peak seen this gesture (screens/sec), used
    /// to detect flick release.
    speed: f32,
    peak: f32,
}

/// The momentum tail after a flick was released: the finger is already lifted
/// and the OS's decaying scroll events are ignored until they stop or you
/// scroll again.
struct Coast {
    /// Speed (screens/sec) of the last tail sample, to spot a re-engagement.
    speed: f32,
    /// Host time of that sample, for the idle reset.
    last_event: f64,
}

pub struct DeviceHubApp {
    frames: FrameSlot,
    status: StatusSlot,
    clipboard: ClipboardSlot,
    /// Current device orientation; drives texture rotation and how pointer
    /// coordinates map back into the device's touch space.
    orientation: OrientationSlot,
    /// Latest clipboard event plus the host time it was first shown, for the
    /// fading indicator.
    clip_activity: Option<(ClipboardEvent, f64)>,
    /// Input commands flow to whichever session is currently live (the manager
    /// swaps the inner channel as it connects/switches devices).
    input: InputSink,
    vnc: VncControl,
    /// Editable VNC bind-host buffer; committed to `vnc` when the server is
    /// enabled.
    vnc_host: String,
    /// Editable VNC bind-port buffer; committed to `vnc` when the server is
    /// enabled.
    vnc_port: String,
    /// Editable VNC password buffer (empty = no auth); committed to `vnc` when
    /// the server is enabled.
    vnc_password: String,
    device_list: DeviceListSlot,
    active: ActiveSlot,
    /// Why the last session failed, shown next to the picker.
    error: ErrorSlot,
    /// Control channel to the session manager. `Option` so closing it on drop
    /// tells the manager to quit. See [`Self::control`].
    control_tx: Option<UnboundedSender<ControlCmd>>,
    session_thread: Option<std::thread::JoinHandle<()>>,

    texture: Option<egui::TextureHandle>,
    tex_size: [usize; 2],
    /// Frame version currently uploaded into `texture`, so we only re-upload on
    /// a newer one.
    last_frame_version: u64,

    /// Synthetic finger driven by the mouse: `Some(last_pos)` while a button is
    /// held. Planting on press (not release) is what makes a press-and-hold
    /// actually hold on the device.
    mouse_touch: Option<egui::Pos2>,

    /// Hardware button currently held down by the pointer. Like `mouse_touch`,
    /// holding on press is what makes a press-and-hold actually hold.
    held_button: Option<&'static str>,

    scroll: Option<ScrollGesture>,
    coast: Option<Coast>,
}

impl DeviceHubApp {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        frames: FrameSlot,
        status: StatusSlot,
        clipboard: ClipboardSlot,
        orientation: OrientationSlot,
        device_list: DeviceListSlot,
        active: ActiveSlot,
        error: ErrorSlot,
        input: InputSink,
        vnc: VncControl,
        control_tx: UnboundedSender<ControlCmd>,
        session_thread: std::thread::JoinHandle<()>,
    ) -> Self {
        // Seed the editable host/port fields from the control's initial bind
        // address (loaded from persisted settings, falling back to loopback).
        let addr = vnc.addr();
        let (vnc_host, vnc_port) = match addr.rsplit_once(':') {
            Some((host, port)) => (host.to_string(), port.to_string()),
            None => ("127.0.0.1".to_string(), "5900".to_string()),
        };
        let vnc_password = vnc.password();
        Self {
            frames,
            status,
            clipboard,
            orientation,
            clip_activity: None,
            input,
            vnc,
            vnc_host,
            vnc_port,
            vnc_password,
            device_list,
            active,
            error,
            control_tx: Some(control_tx),
            session_thread: Some(session_thread),
            texture: None,
            tex_size: [0, 0],
            last_frame_version: 0,
            mouse_touch: None,
            held_button: None,
            scroll: None,
            coast: None,
        }
    }

    fn send(&self, cmd: InputCmd) {
        self.input.send(cmd);
    }

    /// Send a control command to the session manager (switch/refresh device).
    fn control(&self, cmd: ControlCmd) {
        if let Some(tx) = &self.control_tx {
            let _ = tx.send(cmd);
        }
    }

    /// Draw the device picker; selecting a device tells the manager to connect.
    fn device_picker(&self, ui: &mut egui::Ui) {
        let devices = self.device_list.get();
        let active = self.active.get();

        let selected_label = match active
            .as_ref()
            .and_then(|udid| devices.iter().find(|d| &d.udid == udid))
        {
            Some(d) => format!("{} ({})", d.name, d.connection.label()),
            None if active.is_some() => "connecting...".to_owned(),
            None => "Select device...".to_owned(),
        };

        ui.horizontal(|ui| {
            egui::ComboBox::from_id_salt("device_picker")
                .selected_text(selected_label)
                .width(ui.available_width() - 28.0)
                .show_ui(ui, |ui| {
                    if devices.is_empty() {
                        ui.label("no devices connected");
                    }
                    for dev in &devices {
                        let is_active = active.as_deref() == Some(dev.udid.as_str());
                        let label = format!("{} · {}", dev.name, dev.connection.label());
                        if ui
                            .selectable_label(is_active, label)
                            .on_hover_text(&dev.udid)
                            .clicked()
                            && !is_active
                        {
                            self.control(ControlCmd::Connect(dev.udid.clone()));
                        }
                    }
                });
            if ui.button("⟳").on_hover_text("Refresh devices").clicked() {
                self.control(ControlCmd::Refresh);
            }
        });

        if let Some(udid) = &active {
            ui.small(udid);
        }

        if let Some(message) = self.error.get() {
            ui.add_space(2.0);
            ui.colored_label(ui.visuals().error_fg_color, format!("⚠ {message}"));
        }
    }

    /// Draw the VNC server controls. Fields are locked while the server runs so
    /// the displayed values match the live server; enabling commits them.
    fn vnc_controls(&mut self, ui: &mut egui::Ui) {
        ui.label("VNC server");
        let enabled = self.vnc.enabled();

        ui.horizontal(|ui| {
            ui.label("Host");
            ui.add_enabled(
                !enabled,
                egui::TextEdit::singleline(&mut self.vnc_host).desired_width(96.0),
            )
            .on_hover_text("127.0.0.1 for this machine only; 0.0.0.0 to allow other machines");
            ui.label("Port");
            ui.add_enabled(
                !enabled,
                egui::TextEdit::singleline(&mut self.vnc_port).desired_width(56.0),
            );
        });

        ui.horizontal(|ui| {
            ui.label("Password");
            ui.add_enabled(
                !enabled,
                egui::TextEdit::singleline(&mut self.vnc_password)
                    .password(true)
                    .desired_width(120.0),
            )
            .on_hover_text("Leave empty for no authentication (loopback only is safest)");
        });

        let mut want = enabled;
        if ui
            .checkbox(&mut want, "Enable")
            .on_hover_text("Serve this device to standard VNC clients")
            .changed()
        {
            if want {
                match self.vnc_port.trim().parse::<u16>() {
                    Ok(port) => {
                        let host = self.vnc_host.trim();
                        let host = if host.is_empty() { "127.0.0.1" } else { host };
                        self.vnc.set_addr(format!("{host}:{port}"));
                        self.vnc.set_password(self.vnc_password.clone());
                        self.vnc.set_enabled(true);
                    }
                    Err(_) => self.vnc.set_status("invalid port"),
                }
            } else {
                self.vnc.set_enabled(false);
            }
        }

        let status = self.vnc.status();
        if !status.is_empty() {
            ui.small(status);
        }
        let clients = self.vnc.clients();
        if clients > 0 {
            ui.small(format!(
                "{clients} client{} connected",
                if clients == 1 { "" } else { "s" }
            ));
        }
    }

    /// Upload the newest decoded frame into the texture, skipping unchanged frames.
    fn pull_frame(&mut self, ctx: &egui::Context) {
        if let Some((version, frame)) = self.frames.latest() {
            if version == self.last_frame_version {
                return;
            }
            self.last_frame_version = version;
            let size = [frame.width, frame.height];
            let pixels = bytemuck::cast_slice::<u8, egui::Color32>(&frame.rgba).to_vec();
            let image = egui::ColorImage { size, pixels };
            match &mut self.texture {
                Some(tex) if self.tex_size == size => {
                    tex.set(image, egui::TextureOptions::LINEAR);
                }
                _ => {
                    self.texture =
                        Some(ctx.load_texture("screen", image, egui::TextureOptions::LINEAR));
                    self.tex_size = size;
                }
            }
        }
    }

    /// Pick up the latest clipboard event and (re)start the indicator timer.
    fn pull_clipboard(&mut self, ctx: &egui::Context) {
        if let Some(event) = self.clipboard.take() {
            self.clip_activity = Some((event, ctx.input(|i| i.time)));
        }
    }

    /// Draw the transient clipboard-activity indicator, fading out over
    /// [`CLIPBOARD_INDICATOR_TTL`].
    fn clipboard_indicator(&mut self, ui: &mut egui::Ui) {
        let now = ui.input(|i| i.time);
        let Some((event, shown_at)) = &self.clip_activity else {
            return;
        };
        let age = now - shown_at;
        // Hold full opacity for most of the TTL, then fade over the last second.
        let alpha = (1.0 - (age - (CLIPBOARD_INDICATOR_TTL - 1.0)).max(0.0)).clamp(0.0, 1.0) as f32;
        if alpha <= 0.0 {
            self.clip_activity = None;
            return;
        }

        let (arrow, heading) = if event.from_device {
            ("⬇", "Copied from device")
        } else {
            ("⬆", "Sent to device")
        };
        let preview = event.preview.clone();

        egui::Frame::group(ui.style())
            .fill(ui.visuals().faint_bg_color.gamma_multiply(alpha))
            .show(ui, |ui| {
                ui.set_width(ui.available_width());
                let text_color = ui.visuals().text_color().gamma_multiply(alpha);
                let weak_color = ui.visuals().weak_text_color().gamma_multiply(alpha);
                ui.horizontal(|ui| {
                    ui.colored_label(text_color, arrow);
                    ui.colored_label(text_color, "📋");
                    ui.colored_label(text_color, heading);
                });
                if !preview.is_empty() {
                    ui.colored_label(weak_color, preview);
                }
            });
    }

    /// Forward keyboard events to the device.
    fn handle_keyboard(&self, ctx: &egui::Context) {
        ctx.input(|i| {
            for event in &i.events {
                match event {
                    egui::Event::Text(t) => {
                        // Only printable text; control keys come through as Key
                        // events below (avoids double-sending Enter/Tab).
                        let printable: String = t.chars().filter(|c| (*c as u32) >= 0x20).collect();
                        if !printable.is_empty() {
                            self.send(InputCmd::Text(printable));
                        }
                    }
                    egui::Event::Key {
                        key,
                        pressed: true,
                        modifiers,
                        ..
                    } => {
                        // ⌘/⌃/⌥ held -> forward as a chord; the OS suppresses
                        // the Text event for these. Shift alone stays on the
                        // Text path.
                        if modifiers.command || modifiers.ctrl || modifiers.alt {
                            if let Some(usage) = combo_key_usage(*key) {
                                let mods = KeyMods {
                                    cmd: modifiers.command,
                                    shift: modifiers.shift,
                                    ctrl: modifiers.ctrl,
                                    alt: modifiers.alt,
                                };
                                self.send(InputCmd::KeyCombo { usage, mods });
                            }
                        } else if let Some(usage) = special_key_usage(*key) {
                            self.send(InputCmd::KeyUsage(usage));
                        }
                    }
                    _ => {}
                }
            }
        });
    }

    /// Draw the screen texture and handle pointer (tap/drag) + scroll (swipe).
    fn handle_screen(&mut self, ui: &mut egui::Ui, ctx: &egui::Context) {
        let Some(tex) = &self.texture else {
            ui.centered_and_justified(|ui| {
                ui.label("waiting for video...");
            });
            return;
        };

        // Clockwise quarter-turns to apply to the native-portrait frame so the
        // device's orientation shows upright.
        let turns = self.orientation.get().quarter_turns_cw();

        // Fit into the available area, preserving aspect ratio. An odd number of
        // turns is landscape, so the display aspect is the texture's swapped.
        let max = ui.max_rect();
        let (iw, ih) = (self.tex_size[0] as f32, self.tex_size[1] as f32);
        if iw <= 0.0 || ih <= 0.0 {
            return;
        }
        let (dw, dh) = if turns % 2 == 1 { (ih, iw) } else { (iw, ih) };
        let scale = (max.width() / dw).min(max.height() / dh);
        let size = egui::vec2(dw * scale, dh * scale);
        let rect = egui::Rect::from_center_size(max.center(), size);

        let resp = ui.allocate_rect(rect, egui::Sense::click_and_drag());
        // Paint as a rotated quad: keep the rect axis-aligned and rotate the UVs
        // into the native texture. Each corner samples its `unrotate_norm` point.
        let uv = |dx: f32, dy: f32| {
            let (u, v) = unrotate_norm(dx, dy, turns);
            egui::pos2(u, v)
        };
        let mut mesh = egui::Mesh::with_texture(tex.id());
        let corners = [
            (rect.left_top(), uv(0.0, 0.0)),
            (rect.right_top(), uv(1.0, 0.0)),
            (rect.right_bottom(), uv(1.0, 1.0)),
            (rect.left_bottom(), uv(0.0, 1.0)),
        ];
        for (pos, uv) in corners {
            mesh.vertices.push(egui::epaint::Vertex {
                pos,
                uv,
                color: egui::Color32::WHITE,
            });
        }
        mesh.indices.extend_from_slice(&[0, 1, 2, 0, 2, 3]);
        ui.painter().add(egui::Shape::mesh(mesh));

        // Screen point -> normalized device-touch coordinate: fraction across
        // the upright rect, then inverse-rotated into native framebuffer space.
        let to_norm = |p: egui::Pos2| -> (u16, u16) {
            let dx = (p.x - rect.left()) / rect.width();
            let dy = (p.y - rect.top()) / rect.height();
            let (nx, ny) = unrotate_norm(dx, dy, turns);
            (norm(nx), norm(ny))
        };

        // Pointer -> live touch phases, driven off the raw button-down state so
        // a press-and-hold plants the finger and keeps it planted (what makes a
        // long-press actually hold).
        let pointer_down = resp.is_pointer_button_down_on();
        let pointer_pos = resp.interact_pointer_pos().map(|p| clamp_to_rect(p, rect));
        let mut handled_release = false;
        let mut cmds: Vec<InputCmd> = Vec::new();
        match (self.mouse_touch, pointer_down) {
            (None, true) => {
                if let Some(p) = pointer_pos {
                    let (x, y) = to_norm(p);
                    cmds.push(InputCmd::TouchDown { x, y });
                    self.mouse_touch = Some(p);
                }
            }
            (Some(last), true) => {
                if let Some(p) = pointer_pos
                    && p != last
                {
                    let (x, y) = to_norm(p);
                    cmds.push(InputCmd::TouchMove { x, y });
                    self.mouse_touch = Some(p);
                }
            }
            (Some(last), false) => {
                let (x, y) = to_norm(last);
                cmds.push(InputCmd::TouchUp { x, y });
                self.mouse_touch = None;
                handled_release = true;
            }
            (None, false) => {}
        }
        // A click too fast for the live path to observe (press+release between
        // frames) plants no finger above, catch it as a discrete tap.
        // Suppressed after a release we handled, so press/hold/drag never
        // double-fires.
        if resp.clicked()
            && !handled_release
            && let Some(p) = pointer_pos
        {
            let (x, y) = to_norm(p);
            cmds.push(InputCmd::Tap { x, y });
        }
        for cmd in cmds {
            self.send(cmd);
        }

        // Two-finger scroll -> continuous touch gesture. Use the *raw*
        // (unsmoothed) scroll delta; egui's smoothing adds a laggy momentum tail
        // that doesn't match your fingers. Suppressed mid-drag.
        let (scroll, now, cursor) =
            ctx.input(|i| (i.raw_scroll_delta, i.time, i.pointer.latest_pos()));
        let mouse_dragging = self.mouse_touch.is_some();
        // Diagonal of the on-screen frame, for measuring speed in screens/sec.
        let diag = rect.size().length().max(1.0);

        if !mouse_dragging && resp.hovered() && scroll != egui::Vec2::ZERO {
            let step = egui::vec2(scroll.x, scroll.y) * SCROLL_GAIN * SCROLL_INVERT;

            // Instantaneous speed of this sample (screens/sec).
            let prev_t = self
                .scroll
                .as_ref()
                .map(|g| g.last_event)
                .or(self.coast.as_ref().map(|c| c.last_event))
                .unwrap_or(now);
            let dt = (now - prev_t).max(1.0 / 240.0) as f32;
            let inst = (step.length() / diag) / dt;

            // Coasting on a momentum tail: ignore it unless this sample is
            // faster than the tail (a fresh scroll), which cancels the coast.
            if let Some(c) = &self.coast {
                if inst > c.speed * REENGAGE_RATIO && inst > STOP_SPEED {
                    self.coast = None;
                } else {
                    self.coast = Some(Coast {
                        speed: inst,
                        last_event: now,
                    });
                }
            }

            if self.coast.is_none() {
                // Collect commands and send after the `self.scroll` borrow ends.
                let mut cmds: Vec<InputCmd> = Vec::new();
                match &mut self.scroll {
                    None => {
                        // Arm at the cursor without planting; dt is meaningless
                        // at start so ignore this sample's speed.
                        let start = cursor.unwrap_or(rect.center());
                        self.scroll = Some(ScrollGesture {
                            start,
                            pos: start,
                            last_event: now,
                            pending: step,
                            down: false,
                            speed: 0.0,
                            peak: 0.0,
                        });
                    }
                    Some(g) => {
                        g.last_event = now;
                        // Light smoothing so one noisy sample doesn't read as a
                        // release.
                        g.speed = 0.5 * g.speed + 0.5 * inst;
                        g.peak = g.peak.max(g.speed);
                        if g.down {
                            g.pos = clamp_to_rect(g.pos + step, rect);
                            let (x, y) = to_norm(g.pos);
                            cmds.push(InputCmd::TouchMove { x, y });
                            // Lift on a slowing flick (-> iOS inertia) or a
                            // stopped scroll, then coast.
                            let flick_released =
                                g.peak > FLICK_SPEED && g.speed < g.peak * FLICK_RELEASE_RATIO;
                            if flick_released || g.speed < STOP_SPEED {
                                cmds.push(InputCmd::TouchUp { x, y });
                                self.coast = Some(Coast {
                                    speed: g.speed,
                                    last_event: now,
                                });
                                self.scroll = None;
                            }
                        } else {
                            // Still in the deadzone: accumulate until it's
                            // unmistakably a scroll, then plant and replay the
                            // travel so the device sees real motion.
                            g.pending += step;
                            if g.pending.length() > SCROLL_DEADZONE {
                                let (sx, sy) = to_norm(g.start);
                                cmds.push(InputCmd::TouchDown { x: sx, y: sy });
                                g.pos = clamp_to_rect(g.start + g.pending, rect);
                                let (x, y) = to_norm(g.pos);
                                cmds.push(InputCmd::TouchMove { x, y });
                                g.down = true;
                            }
                        }
                    }
                }
                for cmd in cmds {
                    self.send(cmd);
                }
            }
        }

        // Idle fallback for a scroll that stops dead with no momentum: lift any
        // planted finger.
        if let Some(g) = &self.scroll
            && now - g.last_event > SCROLL_END_IDLE
        {
            if g.down {
                let (x, y) = to_norm(g.pos);
                self.send(InputCmd::TouchUp { x, y });
            }
            self.scroll = None;
        }
        // Drop a finished momentum tail so the next scroll starts clean.
        if let Some(c) = &self.coast
            && now - c.last_event > SCROLL_END_IDLE
        {
            self.coast = None;
        }
    }
}

impl eframe::App for DeviceHubApp {
    /// Persist the VNC settings. Keys must match the load in
    /// [`crate::vnc_settings`].
    fn save(&mut self, storage: &mut dyn eframe::Storage) {
        eframe::set_value(storage, "vnc_host", &self.vnc_host);
        eframe::set_value(storage, "vnc_port", &self.vnc_port);
        eframe::set_value(storage, "vnc_password", &self.vnc_password);
        eframe::set_value(storage, "vnc_enabled", &self.vnc.enabled());
    }

    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        self.pull_frame(ctx);
        self.pull_clipboard(ctx);
        self.handle_keyboard(ctx);

        egui::SidePanel::right("controls")
            .resizable(false)
            .default_width(150.0)
            .show(ctx, |ui| {
                ui.add_space(8.0);
                ui.heading("Device Hub");
                ui.separator();

                ui.label("Device");
                self.device_picker(ui);
                ui.separator();

                ui.label("Hardware buttons");
                for &(name, ..) in NAMED_BUTTONS {
                    let resp = ui.button(pretty_button(name));
                    // Drive off the raw button-down state so a press-and-hold
                    // holds the button until release. Only one button is held at
                    // a time (one pointer).
                    let down = resp.is_pointer_button_down_on();
                    let mut handled_release = false;
                    match (self.held_button == Some(name), down) {
                        (false, true) if self.held_button.is_none() => {
                            self.send(InputCmd::ButtonDown(name));
                            self.held_button = Some(name);
                        }
                        (true, false) => {
                            self.send(InputCmd::ButtonUp(name));
                            self.held_button = None;
                            handled_release = true;
                        }
                        _ => {}
                    }
                    // A click too fast for the held path to observe (press+
                    // release between frames) — catch it as a discrete press.
                    if resp.clicked() && !handled_release {
                        self.send(InputCmd::Button(name));
                    }
                }

                ui.separator();
                ui.label("Rotate");
                ui.horizontal(|ui| {
                    if ui.button("⟲ Left").clicked() {
                        self.send(InputCmd::Rotate(RotateDir::Left));
                    }
                    if ui.button("⟳ Right").clicked() {
                        self.send(InputCmd::Rotate(RotateDir::Right));
                    }
                });

                ui.separator();
                self.vnc_controls(ui);

                ui.separator();
                ui.small("Click = tap");
                ui.small("Drag = touch drag");
                ui.small("Two-finger scroll = swipe");
                ui.small("Type to send keys");
                ui.small("Clipboard syncs both ways");

                self.clipboard_indicator(ui);
            });

        egui::TopBottomPanel::bottom("status").show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.small(self.status.get());
            });
        });

        egui::CentralPanel::default().show(ctx, |ui| {
            self.handle_screen(ui, ctx);
        });

        // Only self-tick while something animates between input/frame events (a
        // scroll/coast gesture's idle-lift and momentum timers, or the fading
        // clipboard indicator). The decode task and egui's input handling drive
        // all other repaints, so a static screen goes idle instead of pinning
        // WindowServer at 60 fps.
        let animating =
            self.scroll.is_some() || self.coast.is_some() || self.clip_activity.is_some();
        if animating {
            ctx.request_repaint_after(std::time::Duration::from_millis(16));
        }
    }
}

impl Drop for DeviceHubApp {
    fn drop(&mut self) {
        // Closing the control channel tells the manager to stop the live session
        // cleanly and exit rather than reconnect.
        self.control_tx.take();
        if let Some(handle) = self.session_thread.take() {
            let _ = handle.join();
        }
    }
}

/// Clamp a point to within `rect` (keeps the synthetic finger on-screen).
fn clamp_to_rect(p: egui::Pos2, rect: egui::Rect) -> egui::Pos2 {
    egui::pos2(
        p.x.clamp(rect.left(), rect.right()),
        p.y.clamp(rect.top(), rect.bottom()),
    )
}

/// Map a non-text egui key to a HID Keyboard/Keypad usage. Returns `None` for
/// keys that arrive as `Event::Text` (handled there) or that we don't forward.
fn special_key_usage(key: egui::Key) -> Option<u64> {
    Some(match key {
        egui::Key::Enter => 0x28,
        egui::Key::Escape => 0x29,
        egui::Key::Backspace => 0x2A,
        egui::Key::Tab => 0x2B,
        egui::Key::Delete => 0x4C,
        egui::Key::ArrowRight => 0x4F,
        egui::Key::ArrowLeft => 0x50,
        egui::Key::ArrowDown => 0x51,
        egui::Key::ArrowUp => 0x52,
        egui::Key::Home => 0x4A,
        egui::Key::End => 0x4D,
        egui::Key::PageUp => 0x4B,
        egui::Key::PageDown => 0x4E,
        _ => return None,
    })
}

/// Map an egui key to a HID Keyboard/Keypad usage for a modifier chord (⌘ C,
/// ⌘ Space...). Unlike [`special_key_usage`], this also covers letters, digits,
/// and space, because the OS suppresses the `Text` event when ⌘/⌃/⌥ is held, so
/// those characters arrive only as `Key` events here. Falls back to the special
/// keys so combos like ⌘← or ⌘⌫ work too.
fn combo_key_usage(key: egui::Key) -> Option<u64> {
    use egui::Key::*;
    Some(match key {
        // Letters a-z -> HID 0x04..=0x1D (contiguous, in alphabetical order).
        A | B | C | D | E | F | G | H | I | J | K | L | M | N | O | P | Q | R | S | T | U | V
        | W | X | Y | Z => 0x04 + (key as u64 - A as u64),
        // Digits 1-9 -> 0x1E..=0x26, then 0 -> 0x27.
        Num1 | Num2 | Num3 | Num4 | Num5 | Num6 | Num7 | Num8 | Num9 => {
            0x1E + (key as u64 - Num1 as u64)
        }
        Num0 => 0x27,
        Space => 0x2C,
        Minus => 0x2D,
        Equals => 0x2E,
        _ => return special_key_usage(key),
    })
}

/// Title-case a button id for display ("volume-up" -> "Volume Up").
fn pretty_button(name: &str) -> String {
    name.split('-')
        .map(|w| {
            let mut c = w.chars();
            match c.next() {
                Some(f) => f.to_uppercase().collect::<String>() + c.as_str(),
                None => String::new(),
            }
        })
        .collect::<Vec<_>>()
        .join(" ")
}
