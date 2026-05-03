pub mod env;
pub mod nes;

use nes_core::core::screen::Screen;

use sdl2::event::Event;
use sdl2::keyboard::Keycode;

/// Lazy-initialized SDL2 display for human rendering.
pub struct Display {
    screen: Screen,
    event_pump: sdl2::EventPump,
}

impl Display {
    fn new(width: usize, height: usize) -> Self {
        let sdl = sdl2::init().expect("Failed to initialize SDL2");
        let screen = Screen::new(&sdl, width, height);
        let event_pump = sdl.event_pump().expect("Failed to get event pump");
        Self { screen, event_pump }
    }

    pub fn present(&mut self, buffer: &[u32]) {
        self.screen.present(buffer);
    }

    pub fn poll_quit(&mut self) -> bool {
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
}

/// Ensure the display is initialized, creating it on first use.
pub fn ensure_display(display: &mut Option<Display>, width: usize, height: usize) -> &mut Display {
    display.get_or_insert_with(|| Display::new(width, height))
}
