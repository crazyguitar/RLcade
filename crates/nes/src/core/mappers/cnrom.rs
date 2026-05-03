use super::mapper::{Mapper, PRG_BANK_MASK, PRG_BANK_SIZE, PRG_ROM_START};
use crate::core::cartridge::Cartridge;
use crate::core::common::{NesAddr, NesByte};
use crate::core::log::*;

// CNROM (Mapper 3)
// ref: https://www.nesdev.org/wiki/CNROM
//
// CPU $8000-$FFFF: 32KB unbanked PRG ROM (fixed)
// PPU $0000-$1FFF: 8KB switchable CHR ROM bank
//
// Bank select register ($8000-$FFFF):
//   bits 0-1: select 8KB CHR bank (up to 32KB CHR)

const CHR_BANK_SELECT_MASK: NesByte = 0x3; // 2-bit: up to 4 banks (32KB)
const CHR_BANK_SHIFT: u16 = 13; // 8KB = 2^13

pub struct Cnrom {
    cartridge: Cartridge,
    /// Whether there is only 1 PRG ROM bank (16KB mirrored)
    is_one_bank: bool,
    /// CHR ROM bank select (2-bit, selects 8KB CHR bank)
    select_chr: NesAddr,
}

impl Cnrom {
    pub fn new(cartridge: Cartridge) -> Self {
        let is_one_bank = cartridge.prg_rom().len() == PRG_BANK_SIZE;
        Self {
            cartridge,
            is_one_bank,
            select_chr: 0,
        }
    }
}

impl Mapper for Cnrom {
    fn cartridge(&self) -> &Cartridge {
        &self.cartridge
    }

    fn has_extended_ram(&self) -> bool {
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
        self.select_chr = (value & CHR_BANK_SELECT_MASK) as NesAddr;
    }

    #[inline]
    fn read_chr(&self, addr: NesAddr) -> NesByte {
        let index = (addr | (self.select_chr << CHR_BANK_SHIFT)) as usize;
        let chr = self.cartridge.chr_rom();
        chr[index % chr.len()]
    }

    #[inline]
    fn write_chr(&mut self, addr: NesAddr, value: NesByte) {
        nes_warn!("Read-only CHR memory write attempt at {:#06X}", addr);
    }
}
