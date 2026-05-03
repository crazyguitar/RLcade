use super::bus::{Bus, CpuBus, PpuBus};
use super::cartridge::Cartridge;
use super::common::{NesByte, NesPixel};
use super::cpu::Cpu;
use super::joypad::Joypad;
use super::log::*;
use super::mappers::{Cnrom, MapperKind, Nrom, Sxrom, Txrom, Uxrom};
use super::ppu::Ppu;
use std::io;

/// PPU runs at 3x CPU clock speed
const PPU_CYCLES_PER_CPU_CYCLE: u32 = 3;

/// Screen dimensions
const SCREEN_WIDTH: usize = 256;
const SCREEN_HEIGHT: usize = 240;

pub struct Emulator {
    cpu: Cpu,
    /// When true, APU clocking is skipped entirely (no audio, no APU IRQs).
    /// Speeds up headless / training runs that only need video frames.
    silent: bool,
}

impl Default for Emulator {
    fn default() -> Self {
        Self::new()
    }
}
impl Emulator {
    pub fn width(&self) -> usize {
        SCREEN_WIDTH
    }
    pub fn height(&self) -> usize {
        SCREEN_HEIGHT
    }
    /// Access the 2KB internal RAM.
    pub fn ram(&self) -> &[u8] {
        self.cpu.bus.ram()
    }

    /// Mutable access to the 2KB internal RAM.
    pub fn ram_mut(&mut self) -> &mut [u8] {
        self.cpu.bus.ram_mut()
    }
    pub fn new() -> Self {
        let ppu_bus = PpuBus::new();
        let ppu = Ppu::new(ppu_bus);
        let joypads = [Joypad::new(), Joypad::new()];
        let cpu_bus = CpuBus::new(ppu, joypads);
        let cpu = Cpu::new(cpu_bus);
        Self { cpu, silent: false }
    }

    /// Mute the emulator — skip all APU processing (~30K ticks/frame saved).
    pub fn mute(&mut self) {
        self.silent = true;
    }

    /// Load a ROM file and set up the mapper
    pub fn load(&mut self, path: &str) -> io::Result<()> {
        let mut cartridge = Cartridge::new();
        cartridge.load(path)?;
        self.init_mapper(cartridge)
    }

    /// Load a ROM from raw bytes and set up the mapper
    pub fn load_bytes(&mut self, data: &[u8]) -> io::Result<()> {
        let mut cartridge = Cartridge::new();
        cartridge.load_bytes(data)?;
        self.init_mapper(cartridge)
    }

    fn init_mapper(&mut self, cartridge: Cartridge) -> io::Result<()> {
        let mapper = self.create_mapper(cartridge)?;
        self.cpu.bus.set_mapper(mapper);
        Ok(())
    }

    /// Create the appropriate mapper for the cartridge
    fn create_mapper(&self, cartridge: Cartridge) -> io::Result<MapperKind> {
        let mapper_number = cartridge.mapper_number();
        match mapper_number {
            0 => Ok(MapperKind::Nrom(Nrom::new(cartridge))),
            1 => Ok(MapperKind::Sxrom(Sxrom::new(cartridge))),
            2 => Ok(MapperKind::Uxrom(Uxrom::new(cartridge))),
            3 => Ok(MapperKind::Cnrom(Cnrom::new(cartridge))),
            4 => Ok(MapperKind::Txrom(Txrom::new(cartridge))),
            _ => {
                nes_warn!("Unsupported mapper: {}", mapper_number);
                Err(io::Error::new(
                    io::ErrorKind::Unsupported,
                    format!("Unsupported mapper: {}", mapper_number),
                ))
            }
        }
    }

    /// Reset the emulator. Restores CPU, PPU, and mapper state so successive
    /// episodes start from an identical power-on configuration.
    pub fn reset(&mut self) {
        self.cpu.reset();
        self.cpu.bus.ppu_mut().reset();
        if let Some(m) = self.cpu.bus.mapper_mut() {
            m.reset();
        }
    }

    /// Execute one CPU instruction and run the corresponding PPU and APU cycles.
    /// Returns true if a frame has completed (vblank started).
    #[inline]
    pub fn step(&mut self) -> bool {
        let cpu_cycles = self.cpu.step();
        self.clock_ppu(cpu_cycles);
        self.clock_apu(cpu_cycles);

        // Interrupt checks — order matters: NMI > mapper IRQ > APU IRQ
        self.check_nmi();
        self.check_mapper_irq();
        self.check_apu_irq();

        self.is_frame_complete()
    }

    /// Clock the PPU for the given CPU cycles (3 PPU cycles per CPU cycle).
    #[inline]
    fn clock_ppu(&mut self, cpu_cycles: u32) {
        for _ in 0..cpu_cycles * PPU_CYCLES_PER_CPU_CYCLE {
            let scanline_irq = self.cpu.bus.ppu_mut().cycle();

            // Clock mapper scanline counter on A12 rising edge
            if scanline_irq && let Some(m) = self.cpu.bus.mapper_mut() {
                m.scanline_counter();
            }
        }
    }

    /// Clock APU channels and fetch DMC samples for the given CPU cycles.
    #[inline]
    fn clock_apu(&mut self, cpu_cycles: u32) {
        if self.silent {
            return;
        }
        for _ in 0..cpu_cycles {
            self.cpu.bus.apu.clock();

            if self.cpu.bus.apu.dmc_needs_sample() {
                let addr = self.cpu.bus.apu.dmc_sample_address();
                let sample = self.cpu.bus.read(addr);
                self.cpu.bus.apu.dmc_load_sample(sample);
            }
        }
    }

    /// Check for NMI from PPU (vblank).
    #[inline]
    fn check_nmi(&mut self) {
        if self.cpu.bus.ppu().nmi_pending {
            self.cpu.bus.ppu_mut().nmi_pending = false;
            self.cpu.nmi();
        }
    }

    /// Check for mapper IRQ (e.g. MMC3 scanline counter) and mirroring changes.
    #[inline]
    fn check_mapper_irq(&mut self) {
        let (irq, mirroring) = match self.cpu.bus.mapper_mut() {
            Some(m) => (m.take_irq(), m.take_mirroring()),
            None => return,
        };
        if irq {
            self.cpu.irq();
        }
        if let Some(mode) = mirroring {
            self.cpu.bus.ppu_mut().bus.update_mirroring(mode);
        }
    }

    /// Check for APU IRQ (frame counter or DMC).
    #[inline]
    fn check_apu_irq(&mut self) {
        if self.silent {
            return;
        }
        if self.cpu.bus.apu.irq_pending() {
            self.cpu.irq();
        }
    }

    /// Check and consume the PPU frame-complete flag.
    #[inline]
    fn is_frame_complete(&mut self) -> bool {
        let ppu = self.cpu.bus.ppu_mut();
        if ppu.frame_complete {
            ppu.frame_complete = false;
            return true;
        }
        false
    }

    /// Run emulation until a full frame is completed
    pub fn step_frame(&mut self) {
        loop {
            if self.step() {
                break;
            }
        }
    }

    /// Borrow the PPU's framebuffer directly — no allocation, no copy.
    pub fn screen_buffer(&self) -> &[NesPixel] {
        self.cpu.bus.ppu().screen_buffer()
    }

    /// Set joypad button state for a player
    pub fn set_joypad(&mut self, player: usize, state: NesByte) {
        self.cpu.bus.set_joypad(player, state);
    }

    /// Take buffered audio samples from the APU
    pub fn take_audio_samples(&mut self) -> Vec<f32> {
        self.cpu.bus.apu.take_samples()
    }
}
