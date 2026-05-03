use super::mapper::{CHR_RAM_SIZE, Mapper, PRG_BANK_MASK, PRG_BANK_SIZE};
use crate::core::cartridge::Cartridge;
use crate::core::common::{NesAddr, NesByte};
use crate::core::log::*;

// SxROM / MMC1 (Mapper 1)
// ref: https://www.nesdev.org/wiki/MMC1
//
// CPU $8000-$BFFF: 16KB PRG ROM bank (switchable or fixed)
// CPU $C000-$FFFF: 16KB PRG ROM bank (fixed or switchable)
// PPU $0000-$0FFF: 4KB CHR bank (switchable)
// PPU $0000-$1FFF: 4KB CHR bank (switchable)
//
// Control register ($8000-$9FFF):
//   bits 0-1: mirroring mode
//   bits 2-3: PRG ROM bank mode
//   bit 4:    CHR ROM bank mode
//
// Writes to $8000-$FFFF go through a 5-bit serial shift register.

const RESET_BIT: NesByte = 0x80;
const SHIFT_COMPLETE: i32 = 5;
const CHR_BANK_SIZE: usize = 0x1000; // 4KB

// Control register masks
const MIRROR_MASK: NesByte = 0x03;
const CHR_MODE_BIT: NesByte = 0x10;
const CHR_MODE_SHIFT: u8 = 4;
const PRG_MODE_MASK: NesByte = 0x0C;
const PRG_MODE_SHIFT: u8 = 2;

const CHR_8KB_MASK: NesByte = 0xFE; // mask off bit 0 for 8KB CHR mode
const PRG_BANK_SELECT_MASK: NesByte = 0x0F; // 4-bit PRG bank select

pub struct Sxrom {
    cartridge: Cartridge,
    /// Whether the cartridge uses CHR RAM
    has_character_ram: bool,
    /// CHR ROM bank mode (0 = one 8KB bank, 1 = two 4KB banks)
    mode_chr: i32,
    /// PRG ROM bank mode (0-1 = 32KB, 2 = fix first, 3 = fix last)
    mode_prg: i32,
    /// 5-bit shift register for serial writes
    temp_register: NesByte,
    /// Number of bits written to shift register
    write_counter: i32,
    /// PRG bank register
    register_prg: NesByte,
    /// CHR bank 0 register
    register_chr0: NesByte,
    /// CHR bank 1 register
    register_chr1: NesByte,
    /// Offset into PRG ROM for first 16KB bank
    first_bank_prg: usize,
    /// Offset into PRG ROM for second 16KB bank
    second_bank_prg: usize,
    /// Offset into CHR ROM for first 4KB bank
    first_bank_chr: usize,
    /// Offset into CHR ROM for second 4KB bank
    second_bank_chr: usize,
    /// CHR RAM (8KB, used when cartridge has no CHR ROM)
    character_ram: Vec<NesByte>,
    /// Pending mirroring change from control register write
    new_mirroring: Option<u8>,
}

impl Sxrom {
    pub fn new(cartridge: Cartridge) -> Self {
        let has_character_ram = cartridge.chr_rom().is_empty();
        let second_bank_prg = cartridge.prg_rom().len() - PRG_BANK_SIZE;
        let character_ram = if has_character_ram {
            nes_info!("Uses character RAM");
            vec![0; CHR_RAM_SIZE]
        } else {
            Vec::new()
        };
        Self {
            cartridge,
            has_character_ram,
            mode_chr: 0,
            mode_prg: 3,
            temp_register: 0,
            write_counter: 0,
            register_prg: 0,
            register_chr0: 0,
            register_chr1: 0,
            first_bank_prg: 0,
            second_bank_prg,
            first_bank_chr: 0,
            second_bank_chr: 0,
            character_ram,
            new_mirroring: None,
        }
    }

    /// Number of 16KB PRG banks in the cartridge
    fn prg_bank_count(&self) -> usize {
        self.cartridge.prg_rom().len() / PRG_BANK_SIZE
    }

    /// Recalculate PRG ROM bank pointers based on current mode and register
    fn calculate_prg_pointers(&mut self) {
        let count = self.prg_bank_count();
        if self.mode_prg <= 1 {
            // 32KB switchable: select 32KB bank (ignore low bit)
            let bank = (self.register_prg & !1) as usize % count;
            self.first_bank_prg = PRG_BANK_SIZE * bank;
            self.second_bank_prg = PRG_BANK_SIZE * ((bank + 1) % count);
        } else if self.mode_prg == 2 {
            // Fix first bank at $8000, switch second bank
            self.first_bank_prg = 0;
            self.second_bank_prg = PRG_BANK_SIZE * (self.register_prg as usize % count);
        } else {
            // Switch first bank, fix last bank at $C000
            self.first_bank_prg = PRG_BANK_SIZE * (self.register_prg as usize % count);
            self.second_bank_prg = self.cartridge.prg_rom().len() - PRG_BANK_SIZE;
        }
    }

    /// Reset shift register and set PRG mode to 3 (fix last bank)
    fn reset_shift_register(&mut self) {
        self.temp_register = 0;
        self.write_counter = 0;
        self.mode_prg = 3;
        self.calculate_prg_pointers();
    }

    /// Number of 4KB CHR banks in the cartridge
    fn chr_bank_count(&self) -> usize {
        if self.has_character_ram {
            CHR_RAM_SIZE / CHR_BANK_SIZE
        } else {
            self.cartridge.chr_rom().len() / CHR_BANK_SIZE
        }
    }

    /// Recalculate CHR bank pointers based on current mode
    fn calculate_chr_pointers(&mut self) {
        let count = self.chr_bank_count();
        if self.mode_chr == 0 {
            // One 8KB bank: low bit ignored (clear it to select even bank pair)
            let bank = (self.register_chr0 & CHR_8KB_MASK) as usize % count;
            self.first_bank_chr = CHR_BANK_SIZE * bank;
            self.second_bank_chr = CHR_BANK_SIZE * ((bank + 1) % count);
        } else {
            // Two 4KB banks
            self.first_bank_chr = CHR_BANK_SIZE * (self.register_chr0 as usize % count);
            self.second_bank_chr = CHR_BANK_SIZE * (self.register_chr1 as usize % count);
        }
    }

    /// Handle control register write ($8000-$9FFF)
    /// ref: https://www.nesdev.org/wiki/MMC1#Control_(internal,_$8000-$9FFF)
    fn write_control(&mut self, value: NesByte) {
        // Bits 0-1: mirroring mode (0=one-lower, 1=one-upper, 2=vertical, 3=horizontal)
        let mirror_bits = value & MIRROR_MASK;
        let mirroring = match mirror_bits {
            0 => 2, // one-screen lower bank
            1 => 3, // one-screen upper bank
            2 => 1, // vertical
            3 => 0, // horizontal
            _ => unreachable!(),
        };
        self.new_mirroring = Some(mirroring);

        self.mode_chr = ((value & CHR_MODE_BIT) >> CHR_MODE_SHIFT) as i32;
        self.mode_prg = ((value & PRG_MODE_MASK) >> PRG_MODE_SHIFT) as i32;
        self.calculate_prg_pointers();
        self.calculate_chr_pointers();
    }

    /// Handle CHR bank 0 register write ($A000-$BFFF)
    fn write_chr_bank0(&mut self, value: NesByte) {
        self.register_chr0 = value;
        self.calculate_chr_pointers();
    }

    /// Handle CHR bank 1 register write ($C000-$DFFF)
    fn write_chr_bank1(&mut self, value: NesByte) {
        self.register_chr1 = value;
        self.calculate_chr_pointers();
    }

    /// Handle PRG bank register write ($E000-$FFFF).
    ///
    /// Bit 4 of the value toggles PRG-RAM chip enable on real hardware.
    /// Not implemented here: writes to PRG-RAM are always accepted regardless
    /// of this flag, so save-enabled titles that gate SRAM writes via this
    /// bit may behave incorrectly. This is rare in practice for the ROMs
    /// targeted by this emulator (SMB1 uses NROM, not MMC1).
    fn write_prg_bank(&mut self, value: NesByte) {
        self.register_prg = value & PRG_BANK_SELECT_MASK;
        self.calculate_prg_pointers();
    }
}

impl Mapper for Sxrom {
    fn cartridge(&self) -> &Cartridge {
        &self.cartridge
    }

    fn has_extended_ram(&self) -> bool {
        false
    }

    #[inline]
    fn read_prg(&self, addr: NesAddr) -> NesByte {
        let offset = (addr & PRG_BANK_MASK) as usize;
        match addr {
            0x8000..0xC000 => self.cartridge.prg_rom()[self.first_bank_prg + offset],
            0xC000..=0xFFFF => self.cartridge.prg_rom()[self.second_bank_prg + offset],
            _ => 0,
        }
    }

    #[inline]
    fn write_prg(&mut self, addr: NesAddr, value: NesByte) {
        // Reset if bit 7 is set
        if value & RESET_BIT != 0 {
            self.reset_shift_register();
            return;
        }

        // Serial write: shift in bit 0 of value
        self.temp_register = (self.temp_register >> 1) | ((value & 1) << 4);
        self.write_counter += 1;

        if self.write_counter < SHIFT_COMPLETE {
            return;
        }

        // 5 bits collected — decode based on address range
        let value = self.temp_register;
        match addr {
            0x8000..0xA000 => self.write_control(value),
            0xA000..0xC000 => self.write_chr_bank0(value),
            0xC000..0xE000 => self.write_chr_bank1(value),
            0xE000..=0xFFFF => self.write_prg_bank(value),
            _ => {}
        }

        self.temp_register = 0;
        self.write_counter = 0;
    }

    #[inline]
    fn read_chr(&self, addr: NesAddr) -> NesByte {
        if self.has_character_ram {
            return self.character_ram[addr as usize & (CHR_RAM_SIZE - 1)];
        }

        let cartridge = &self.cartridge;
        let first_bank_chr = self.first_bank_chr;
        let second_bank_chr = self.second_bank_chr;
        match addr {
            0x0000..0x1000 => cartridge.chr_rom()[first_bank_chr + addr as usize],
            0x1000..0x2000 => cartridge.chr_rom()[second_bank_chr + (addr & 0x0FFF) as usize],
            _ => 0,
        }
    }

    #[inline]
    fn write_chr(&mut self, addr: NesAddr, value: NesByte) {
        if self.has_character_ram {
            self.character_ram[addr as usize & (CHR_RAM_SIZE - 1)] = value;
        } else {
            nes_warn!("Read-only CHR memory write attempt at {:#06X}", addr);
        }
    }

    #[inline]
    fn take_mirroring(&mut self) -> Option<u8> {
        self.new_mirroring.take()
    }

    fn reset(&mut self) {
        self.mode_chr = 0;
        self.mode_prg = 3;
        self.temp_register = 0;
        self.write_counter = 0;
        self.register_prg = 0;
        self.register_chr0 = 0;
        self.register_chr1 = 0;
        self.first_bank_prg = 0;
        self.second_bank_prg = self.cartridge.prg_rom().len() - PRG_BANK_SIZE;
        self.first_bank_chr = 0;
        self.second_bank_chr = 0;
        self.new_mirroring = None;
        if self.has_character_ram {
            self.character_ram.fill(0);
        }
    }
}
