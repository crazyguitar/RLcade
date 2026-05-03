use super::mapper::{CHR_RAM_SIZE, Mapper, PRG_BANK_MASK, PRG_BANK_SIZE, PRG_ROM_START};
use crate::core::cartridge::Cartridge;
use crate::core::common::{NesAddr, NesByte};
use crate::core::log::*;

// NROM (Mapper 0)
// ref: https://www.nesdev.org/wiki/NROM
//
// CPU $8000-$BFFF: 16KB PRG ROM bank (or mirrored if only 16KB)
// CPU $C000-$FFFF: 16KB PRG ROM bank (or mirror of $8000-$BFFF)
// PPU $0000-$1FFF: 8KB CHR ROM (or CHR RAM if no CHR ROM present)
//
// No bank switching — simplest mapper.

pub struct Nrom {
    cartridge: Cartridge,
    /// Whether there is only 1 PRG ROM bank (16KB mirrored)
    is_one_bank: bool,
    /// Whether this mapper uses CHR RAM instead of CHR ROM
    has_character_ram: bool,
    /// CHR RAM (8KB, used when cartridge has no CHR ROM)
    character_ram: Vec<NesByte>,
}

impl Nrom {
    pub fn new(cartridge: Cartridge) -> Self {
        let is_one_bank = cartridge.prg_rom().len() == PRG_BANK_SIZE;
        let has_character_ram = cartridge.chr_rom().is_empty();
        let character_ram = if has_character_ram {
            nes_info!("Uses character RAM");
            vec![0; CHR_RAM_SIZE]
        } else {
            Vec::new()
        };
        Self {
            cartridge,
            is_one_bank,
            has_character_ram,
            character_ram,
        }
    }
}

impl Mapper for Nrom {
    fn cartridge(&self) -> &Cartridge {
        &self.cartridge
    }

    fn has_extended_ram(&self) -> bool {
        // NOTE: Family BASIC uses PRG-RAM at $6000-$7FFF, not supported
        false
    }

    #[inline]
    fn read_prg(&self, addr: NesAddr) -> NesByte {
        if self.is_one_bank {
            self.cartridge.prg_rom()[((addr - PRG_ROM_START) & PRG_BANK_MASK) as usize]
        } else {
            self.cartridge.prg_rom()[(addr - PRG_ROM_START) as usize]
        }
    }

    #[inline]
    fn write_prg(&mut self, addr: NesAddr, value: NesByte) {
        nes_warn!(
            "ROM memory write attempt at {:#06X} to set {:#04X}",
            addr,
            value
        );
    }

    #[inline]
    fn read_chr(&self, addr: NesAddr) -> NesByte {
        if self.has_character_ram {
            self.character_ram[addr as usize]
        } else {
            self.cartridge.chr_rom()[addr as usize]
        }
    }

    #[inline]
    fn write_chr(&mut self, addr: NesAddr, value: NesByte) {
        if self.has_character_ram {
            self.character_ram[addr as usize] = value;
        } else {
            nes_warn!("Read-only CHR memory write attempt at {:#06X}", addr);
        }
    }
}
