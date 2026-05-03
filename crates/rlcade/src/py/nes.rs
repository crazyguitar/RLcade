//! Python bindings for the raw NES emulator.

use pyo3::prelude::*;
use pyo3::types::{PyByteArray, PyBytes};

use nes_core::core::emulator::Emulator;
use nes_core::core::nes::Nes;
use nes_core::core::screen::to_rgb;

use super::Display;

const NUM_ACTIONS: u16 = 256;

#[pyclass(name = "Screen", unsendable)]
pub struct PyScreen {
    pub rgb: Vec<u8>,
}

#[pymethods]
impl PyScreen {
    fn bytes<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, &self.rgb)
    }

    fn bytearray<'py>(&self, py: Python<'py>) -> Bound<'py, PyByteArray> {
        PyByteArray::new(py, &self.rgb)
    }
}

#[pyclass(name = "Nes", unsendable)]
pub struct PyNes {
    emu: Emulator,
    display: Option<Display>,
}

#[pymethods]
impl PyNes {
    #[new]
    fn new(rom: &str) -> PyResult<Self> {
        let mut emu = Emulator::new();
        emu.load(rom)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
        emu.reset();
        Ok(Self { emu, display: None })
    }

    fn reset(&mut self) {
        self.emu.reset();
    }

    #[pyo3(signature = (p1, p2=0))]
    fn step(&mut self, p1: u8, p2: u8) {
        self.emu.set_joypad(0, p1);
        self.emu.set_joypad(1, p2);
        self.emu.step_frame();
    }

    #[rustfmt::skip]
    fn set_joypad(&mut self, player: usize, action: u8) { self.emu.set_joypad(player, action); }
    #[rustfmt::skip]
    fn mute(&mut self) { self.emu.mute(); }

    fn screen(&self) -> PyScreen {
        let rgb = to_rgb(
            self.emu.screen_buffer(),
            self.emu.width(),
            self.emu.height(),
        );
        PyScreen { rgb }
    }

    fn ram<'py>(&self, py: Python<'py>) -> Bound<'py, PyBytes> {
        PyBytes::new(py, self.emu.ram())
    }

    fn write_ram(&mut self, address: u16, value: u8) {
        self.emu.ram_mut()[address as usize] = value;
    }

    fn render(&mut self) {
        let d = super::ensure_display(&mut self.display, self.emu.width(), self.emu.height());
        d.present(self.emu.screen_buffer());
    }

    fn poll_quit(&mut self) -> bool {
        let d = super::ensure_display(&mut self.display, self.emu.width(), self.emu.height());
        d.poll_quit()
    }

    #[getter]
    fn screen_height(&self) -> usize {
        self.emu.height()
    }
    #[getter]
    fn screen_width(&self) -> usize {
        self.emu.width()
    }
    #[getter]
    fn num_actions(&self) -> u16 {
        NUM_ACTIONS
    }
}

#[pyfunction]
pub fn play(rom: &str) {
    let mut nes = Nes::new(rom);
    nes.run();
}
