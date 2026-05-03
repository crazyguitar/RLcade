use super::audio::Audio;
use super::emulator::Emulator;
use super::joypad::*;
use super::log::*;
use super::screen::Screen;
use sdl2::EventPump;
use sdl2::event::Event;
use sdl2::keyboard::{Keycode, Scancode};
use std::time::{Duration, Instant};

/// NTSC NES runs at ~60 FPS
const FRAME_DURATION: Duration = Duration::from_nanos(1_000_000_000 / 60);

/// NES emulator with SDL2 display, audio, and keyboard input for interactive play.
pub struct Nes {
    emu: Emulator,
    screen: Screen,
    audio: Audio,
    event_pump: EventPump,
}

impl Nes {
    /// Create a new interactive NES instance.
    pub fn new(rom: &str) -> Self {
        let emu = Self::load(rom);
        let sdl = sdl2::init().expect("Failed to initialize SDL2");
        let screen = Screen::new(&sdl, emu.width(), emu.height());
        let audio = Audio::new(&sdl);
        let event_pump = sdl.event_pump().expect("Failed to get event pump");

        Self {
            emu,
            screen,
            audio,
            event_pump,
        }
    }

    /// Load a ROM file into a new Emulator instance, exit on failure.
    fn load(path: &str) -> Emulator {
        let mut emu = Emulator::new();
        if let Err(e) = emu.load(path) {
            eprintln!("Failed to load ROM '{}': {}", path, e);
            std::process::exit(1);
        }
        emu.reset();
        emu
    }

    /// Main emulation loop: poll input, step NES, render and present each frame.
    pub fn run(&mut self) {
        nes_info!("P1: Arrow keys = D-pad, J = A, K = B, M = Start, N = Select");
        nes_info!("P2: WASD = D-pad, G = A, F = B, Y = Start, T = Select");
        nes_info!("Press Escape to quit.");

        loop {
            let frame_start = Instant::now();

            if self.poll_quit() {
                break;
            }

            let p1 = Self::poll_joypad(&self.event_pump, &Self::P1_KEYS);
            let p2 = Self::poll_joypad(&self.event_pump, &Self::P2_KEYS);
            self.emu.set_joypad(PLAYER1, p1);
            self.emu.set_joypad(PLAYER2, p2);
            self.emu.step_frame();
            self.audio.flush(&self.emu.take_audio_samples());
            self.screen.present(self.emu.screen_buffer());

            Self::limit_fps(frame_start);
        }
    }

    /// Consume pending SDL2 events, return true if quit was requested.
    fn poll_quit(&mut self) -> bool {
        for event in self.event_pump.poll_iter() {
            match event {
                Event::Quit { .. }
                | Event::KeyDown {
                    keycode: Some(Keycode::Escape),
                    ..
                } => return true,
                _ => {}
            }
        }
        false
    }

    /// Key bindings for each player: (Scancode, button bitmask) pairs.
    const P1_KEYS: [(Scancode, u8); 8] = [
        (Scancode::J, BUTTON_A),
        (Scancode::K, BUTTON_B),
        (Scancode::N, BUTTON_SELECT),
        (Scancode::M, BUTTON_START),
        (Scancode::Up, BUTTON_UP),
        (Scancode::Down, BUTTON_DOWN),
        (Scancode::Left, BUTTON_LEFT),
        (Scancode::Right, BUTTON_RIGHT),
    ];

    const P2_KEYS: [(Scancode, u8); 8] = [
        (Scancode::G, BUTTON_A),
        (Scancode::F, BUTTON_B),
        (Scancode::T, BUTTON_SELECT),
        (Scancode::Y, BUTTON_START),
        (Scancode::W, BUTTON_UP),
        (Scancode::S, BUTTON_DOWN),
        (Scancode::A, BUTTON_LEFT),
        (Scancode::D, BUTTON_RIGHT),
    ];

    /// Read keyboard state and return NES joypad button bitmask for the given key bindings.
    fn poll_joypad(event_pump: &EventPump, bindings: &[(Scancode, u8)]) -> u8 {
        let keys = event_pump.keyboard_state();
        let mut state: u8 = 0;
        for &(scancode, button) in bindings {
            if keys.is_scancode_pressed(scancode) {
                state |= button;
            }
        }
        state
    }

    /// Sleep for the remainder of the frame to maintain ~60 FPS.
    fn limit_fps(frame_start: Instant) {
        let elapsed = frame_start.elapsed();
        if elapsed < FRAME_DURATION {
            std::thread::sleep(FRAME_DURATION - elapsed);
        }
    }
}
