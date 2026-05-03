use super::mapper::{CHR_RAM_SIZE, Mapper, PRG_BANK_MASK, PRG_BANK_SIZE, PRG_ROM_START};
use crate::core::cartridge::Cartridge;
use crate::core::common::{NesAddr, NesByte};
use crate::core::log::*;

// UxROM (Mapper 2)
// ref: https://www.nesdev.org/wiki/UxROM
//
// CPU $8000-$BFFF: 16KB switchable PRG ROM bank
// CPU $C000-$FFFF: 16KB PRG ROM bank (fixed to last bank)
// PPU $0000-$1FFF: 8KB CHR ROM or CHR RAM
//
// Bank select register ($8000-$FFFF):
//   value selects 16KB PRG ROM bank at $8000-$BFFF

pub struct Uxrom {
    cartridge: Cartridge,
    /// Whether the cartridge uses CHR RAM
    has_character_ram: bool,
    /// Offset into PRG ROM for the fixed last bank
    last_bank_pointer: usize,
    /// PRG ROM bank count (used to wrap `select_prg`)
    prg_bank_count: usize,
    /// PRG ROM bank select, already wrapped into `[0, prg_bank_count)`
    select_prg: usize,
    /// CHR RAM (8KB, used when cartridge has no CHR ROM)
    character_ram: Vec<NesByte>,
}

impl Uxrom {
    pub fn new(cartridge: Cartridge) -> Self {
        let has_character_ram = cartridge.chr_rom().is_empty();
        let prg_bank_count = cartridge.prg_rom().len() / PRG_BANK_SIZE;
        assert!(
            prg_bank_count > 0,
            "UxROM: PRG ROM must have at least one 16KB bank"
        );
        let last_bank_pointer = (prg_bank_count - 1) * PRG_BANK_SIZE;
        let character_ram = if has_character_ram {
            nes_info!("Uses character RAM");
            vec![0; CHR_RAM_SIZE]
        } else {
            Vec::new()
        };
        Self {
            cartridge,
            has_character_ram,
            last_bank_pointer,
            prg_bank_count,
            select_prg: 0,
            character_ram,
        }
    }
}

impl Uxrom {
    /// Read from switchable PRG bank at $8000-$BFFF
    fn read_switchable_bank(&self, addr: NesAddr) -> NesByte {
        let offset =
            self.select_prg * PRG_BANK_SIZE + ((addr - PRG_ROM_START) & PRG_BANK_MASK) as usize;
        self.cartridge.prg_rom()[offset]
    }

    /// Read from fixed last PRG bank at $C000-$FFFF
    fn read_fixed_last_bank(&self, addr: NesAddr) -> NesByte {
        self.cartridge.prg_rom()[self.last_bank_pointer + (addr & PRG_BANK_MASK) as usize]
    }
}

impl Mapper for Uxrom {
    fn cartridge(&self) -> &Cartridge {
        &self.cartridge
    }

    fn has_extended_ram(&self) -> bool {
        false
    }

    #[inline]
    fn read_prg(&self, addr: NesAddr) -> NesByte {
        match addr {
            0x8000..0xC000 => self.read_switchable_bank(addr),
            0xC000..=0xFFFF => self.read_fixed_last_bank(addr),
            _ => 0,
        }
    }

    #[inline]
    fn write_prg(&mut self, _addr: NesAddr, value: NesByte) {
        // Wrap the bank index so a rogue guest write can't produce an OOB PRG
        // read on the next CPU fetch (real hardware implicitly wraps too).
        self.select_prg = (value as usize) % self.prg_bank_count;
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
