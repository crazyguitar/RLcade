use crate::core::cartridge::Cartridge;
use crate::core::common::{NesAddr, NesByte};

// Shared mapper constants
pub const PRG_ROM_START: NesAddr = 0x8000;
pub const PRG_BANK_SIZE: usize = 0x4000; // 16KB
pub const PRG_BANK_MASK: NesAddr = 0x3FFF;
pub const CHR_RAM_SIZE: usize = 0x2000; // 8KB
pub const PRG_BANK_8KB: usize = 0x2000;
pub const PRG_BANK_8KB_MASK: NesAddr = 0x1FFF;
pub const CHR_BANK_1KB: usize = 0x0400;

pub trait Mapper {
    fn cartridge(&self) -> &Cartridge;
    fn has_extended_ram(&self) -> bool;
    fn read_prg(&self, addr: NesAddr) -> NesByte;
    fn write_prg(&mut self, addr: NesAddr, value: NesByte);
    fn read_chr(&self, addr: NesAddr) -> NesByte;
    fn write_chr(&mut self, addr: NesAddr, value: NesByte);

    /// Called on PPU A12 rising edge (used by MMC3 for scanline counting)
    #[inline]
    fn scanline_counter(&mut self) {}

    /// Consume a pending IRQ from the mapper (used by MMC3). Returns true and
    /// clears the internal flag if an IRQ was pending.
    #[inline]
    fn take_irq(&mut self) -> bool {
        false
    }

    /// Consume a pending mirroring change. Returns `Some(mode)` and clears the
    /// internal flag if a new mirroring mode was requested.
    #[inline]
    fn take_mirroring(&mut self) -> Option<u8> {
        None
    }

    /// Restore mapper state to power-on defaults. Called by `Emulator::reset()`
    /// so successive episodes run from an identical mapper configuration.
    /// Mappers that keep no mutable state (e.g. NROM) can leave this as a no-op.
    fn reset(&mut self) {}
}
