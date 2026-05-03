use super::mapper::{CHR_BANK_1KB, CHR_RAM_SIZE, Mapper, PRG_BANK_8KB, PRG_BANK_8KB_MASK};
use crate::core::cartridge::Cartridge;
use crate::core::common::{NesAddr, NesByte};
use crate::core::log::*;

// TxROM / MMC3 (Mapper 4)
// ref: https://www.nesdev.org/wiki/MMC3
//
// PRG ROM: Four 8KB banks at $8000-$FFFF
//   Mode 0: R6 at $8000, R7 at $A000, fixed(-2) at $C000, fixed(-1) at $E000
//   Mode 1: fixed(-2) at $8000, R7 at $A000, R6 at $C000, fixed(-1) at $E000
//
// CHR: Eight 1KB banks at $0000-$1FFF
//   Mode 0: R0 2KB at $0000, R1 2KB at $0800, R2-R5 1KB at $1000-$1FFF
//   Mode 1: R2-R5 1KB at $0000-$0FFF, R0 2KB at $1000, R1 2KB at $1800

const NUM_BANK_REGISTERS: usize = 8;
const NUM_PRG_BANKS: usize = 4;
const NUM_CHR_BANKS: usize = 8;

// Bank select register ($8000) bits
const BANK_SELECT_MASK: NesByte = 0x07;
const PRG_MODE_BIT: NesByte = 0x40;
const CHR_INVERSION_BIT: NesByte = 0x80;

const PRG_BANK_6BIT_MASK: usize = 0x3F; // 6-bit bank select (registers R6, R7)
const CHR_2KB_MASK: usize = 0xFE; // mask off bit 0 for 2KB CHR banks (R0, R1)
const MIRROR_VERTICAL: u8 = 1;
const MIRROR_HORIZONTAL: u8 = 0;

pub struct Txrom {
    cartridge: Cartridge,
    has_character_ram: bool,
    character_ram: Vec<NesByte>,

    /// Bank registers R0-R7
    registers: [NesByte; NUM_BANK_REGISTERS],
    /// Which register (0-7) the next $8001 write targets
    target_register: usize,
    /// PRG banking mode (bit 6 of $8000)
    prg_mode: bool,
    /// CHR A12 inversion (bit 7 of $8000)
    chr_inversion: bool,

    /// Pre-computed PRG bank offsets (4 x 8KB)
    prg_offsets: [usize; NUM_PRG_BANKS],
    /// Pre-computed CHR bank offsets (8 x 1KB)
    chr_offsets: [usize; NUM_CHR_BANKS],

    /// IRQ scanline counter
    irq_counter: u8,
    /// IRQ counter reload value (written via $C000)
    irq_latch: u8,
    /// Flag to reload counter on next A12 rising edge
    irq_reload: bool,
    /// IRQ generation enabled
    irq_enabled: bool,
    /// IRQ pending (needs to be serviced by CPU)
    irq_pending: bool,

    /// Pending mirroring change
    new_mirroring: Option<u8>,
}

impl Txrom {
    pub fn new(cartridge: Cartridge) -> Self {
        let has_character_ram = cartridge.chr_rom().is_empty();
        let character_ram = if has_character_ram {
            nes_info!("MMC3: Uses character RAM");
            vec![0; CHR_RAM_SIZE]
        } else {
            Vec::new()
        };

        let mut mapper = Self {
            cartridge,
            has_character_ram,
            character_ram,
            registers: [0; NUM_BANK_REGISTERS],
            target_register: 0,
            prg_mode: false,
            chr_inversion: false,
            prg_offsets: [0; NUM_PRG_BANKS],
            chr_offsets: [0; NUM_CHR_BANKS],
            irq_counter: 0,
            irq_latch: 0,
            irq_reload: false,
            irq_enabled: false,
            irq_pending: false,
            new_mirroring: None,
        };

        mapper.update_prg_offsets();
        mapper.update_chr_offsets();
        mapper
    }

    /// Number of 8KB PRG banks in the cartridge
    fn prg_bank_count(&self) -> usize {
        self.cartridge.prg_rom().len() / PRG_BANK_8KB
    }

    /// Number of 1KB CHR banks in the cartridge
    fn chr_bank_count(&self) -> usize {
        if self.has_character_ram {
            CHR_RAM_SIZE / CHR_BANK_1KB
        } else {
            self.cartridge.chr_rom().len() / CHR_BANK_1KB
        }
    }

    /// Recalculate PRG ROM bank offsets
    fn update_prg_offsets(&mut self) {
        let count = self.prg_bank_count();
        if count < 2 {
            return;
        }
        let r6 = (self.registers[6] as usize & PRG_BANK_6BIT_MASK) % count;
        let r7 = (self.registers[7] as usize & PRG_BANK_6BIT_MASK) % count;
        let second_last = count - 2;
        let last = count - 1;

        if self.prg_mode {
            // Mode 1: fixed(-2), R7, R6, fixed(-1)
            self.prg_offsets = [
                second_last * PRG_BANK_8KB,
                r7 * PRG_BANK_8KB,
                r6 * PRG_BANK_8KB,
                last * PRG_BANK_8KB,
            ];
        } else {
            // Mode 0: R6, R7, fixed(-2), fixed(-1)
            self.prg_offsets = [
                r6 * PRG_BANK_8KB,
                r7 * PRG_BANK_8KB,
                second_last * PRG_BANK_8KB,
                last * PRG_BANK_8KB,
            ];
        }
    }

    /// Recalculate CHR bank offsets
    fn update_chr_offsets(&mut self) {
        let count = self.chr_bank_count();
        // R0, R1 are 2KB banks (bit 0 ignored)
        let r0 = (self.registers[0] as usize & CHR_2KB_MASK) % count;
        let r1 = (self.registers[1] as usize & CHR_2KB_MASK) % count;
        let r2 = (self.registers[2] as usize) % count;
        let r3 = (self.registers[3] as usize) % count;
        let r4 = (self.registers[4] as usize) % count;
        let r5 = (self.registers[5] as usize) % count;

        if self.chr_inversion {
            // Mode 1: R2-R5 at $0000-$0FFF, R0-R1 at $1000-$1FFF
            self.chr_offsets = [
                r2 * CHR_BANK_1KB,
                r3 * CHR_BANK_1KB,
                r4 * CHR_BANK_1KB,
                r5 * CHR_BANK_1KB,
                r0 * CHR_BANK_1KB,
                ((r0 + 1) % count) * CHR_BANK_1KB,
                r1 * CHR_BANK_1KB,
                ((r1 + 1) % count) * CHR_BANK_1KB,
            ];
        } else {
            // Mode 0: R0-R1 at $0000-$0FFF, R2-R5 at $1000-$1FFF
            self.chr_offsets = [
                r0 * CHR_BANK_1KB,
                ((r0 + 1) % count) * CHR_BANK_1KB,
                r1 * CHR_BANK_1KB,
                ((r1 + 1) % count) * CHR_BANK_1KB,
                r2 * CHR_BANK_1KB,
                r3 * CHR_BANK_1KB,
                r4 * CHR_BANK_1KB,
                r5 * CHR_BANK_1KB,
            ];
        }
    }

    /// $8000 (even): Bank select — choose target register and banking modes
    fn write_bank_select(&mut self, value: NesByte) {
        self.target_register = (value & BANK_SELECT_MASK) as usize;
        let new_prg_mode = value & PRG_MODE_BIT != 0;
        let new_chr_inversion = value & CHR_INVERSION_BIT != 0;

        if self.prg_mode != new_prg_mode {
            self.prg_mode = new_prg_mode;
            self.update_prg_offsets();
        }
        if self.chr_inversion != new_chr_inversion {
            self.chr_inversion = new_chr_inversion;
            self.update_chr_offsets();
        }
    }

    /// $8001 (odd): Bank data — write value to the selected register
    fn write_bank_data(&mut self, value: NesByte) {
        self.registers[self.target_register] = value;
        if self.target_register <= 5 {
            self.update_chr_offsets();
        } else {
            self.update_prg_offsets();
        }
    }

    /// $A000 (even): Mirroring control
    fn write_mirroring(&mut self, value: NesByte) {
        // 0 = vertical, 1 = horizontal
        let mirroring = if value & 1 == 0 {
            MIRROR_VERTICAL
        } else {
            MIRROR_HORIZONTAL
        };
        self.new_mirroring = Some(mirroring);
    }

    /// $C000 (even): Set IRQ counter reload value
    fn write_irq_latch(&mut self, value: NesByte) {
        self.irq_latch = value;
    }

    /// $C001 (odd): Clear counter and flag for reload on next A12 rising edge
    fn write_irq_reload(&mut self) {
        self.irq_counter = 0;
        self.irq_reload = true;
    }

    /// $E000 (even): Disable IRQ and acknowledge any pending IRQ
    fn write_irq_disable(&mut self) {
        self.irq_enabled = false;
        self.irq_pending = false;
    }

    /// $E001 (odd): Enable IRQ generation
    fn write_irq_enable(&mut self) {
        self.irq_enabled = true;
    }
}

impl Mapper for Txrom {
    fn cartridge(&self) -> &Cartridge {
        &self.cartridge
    }

    fn has_extended_ram(&self) -> bool {
        true
    }

    #[inline]
    fn read_prg(&self, addr: NesAddr) -> NesByte {
        let bank = ((addr - 0x8000) / PRG_BANK_8KB as u16) as usize;
        let offset = (addr & PRG_BANK_8KB_MASK) as usize;
        self.cartridge.prg_rom()[self.prg_offsets[bank] + offset]
    }

    #[inline]
    fn write_prg(&mut self, addr: NesAddr, value: NesByte) {
        let is_even = addr & 1 == 0;
        match (addr, is_even) {
            (0x8000..0xA000, true) => self.write_bank_select(value),
            (0x8000..0xA000, false) => self.write_bank_data(value),
            (0xA000..0xC000, true) => self.write_mirroring(value),
            (0xA000..0xC000, false) => { /* RAM protect — not implemented */ }
            (0xC000..0xE000, true) => self.write_irq_latch(value),
            (0xC000..0xE000, false) => self.write_irq_reload(),
            (0xE000..=0xFFFF, true) => self.write_irq_disable(),
            (0xE000..=0xFFFF, false) => self.write_irq_enable(),
            _ => {}
        }
    }

    #[inline]
    fn read_chr(&self, addr: NesAddr) -> NesByte {
        let bank = (addr / CHR_BANK_1KB as u16) as usize;
        let offset = (addr % CHR_BANK_1KB as u16) as usize;
        if self.has_character_ram {
            self.character_ram[self.chr_offsets[bank] + offset]
        } else {
            self.cartridge.chr_rom()[self.chr_offsets[bank] + offset]
        }
    }

    #[inline]
    fn write_chr(&mut self, addr: NesAddr, value: NesByte) {
        if self.has_character_ram {
            let bank = (addr / CHR_BANK_1KB as u16) as usize;
            let offset = (addr % CHR_BANK_1KB as u16) as usize;
            self.character_ram[self.chr_offsets[bank] + offset] = value;
        }
    }

    /// Called on PPU A12 rising edge (once per scanline typically).
    /// IRQ fires on the 1→0 transition only, not when the counter is
    /// reloaded to zero (otherwise `irq_latch == 0` would flood the CPU).
    /// ref: https://www.nesdev.org/wiki/MMC3#IRQ_generation
    #[inline]
    fn scanline_counter(&mut self) {
        if self.irq_counter == 0 || self.irq_reload {
            self.irq_counter = self.irq_latch;
            self.irq_reload = false;
        } else {
            self.irq_counter -= 1;
            if self.irq_counter == 0 && self.irq_enabled {
                self.irq_pending = true;
            }
        }
    }

    #[inline]
    fn take_irq(&mut self) -> bool {
        let pending = self.irq_pending;
        self.irq_pending = false;
        pending
    }

    #[inline]
    fn take_mirroring(&mut self) -> Option<u8> {
        self.new_mirroring.take()
    }

    fn reset(&mut self) {
        self.registers = [0; NUM_BANK_REGISTERS];
        self.target_register = 0;
        self.prg_mode = false;
        self.chr_inversion = false;
        self.irq_counter = 0;
        self.irq_latch = 0;
        self.irq_reload = false;
        self.irq_enabled = false;
        self.irq_pending = false;
        self.new_mirroring = None;
        if self.has_character_ram {
            self.character_ram.fill(0);
        }
        self.update_prg_offsets();
        self.update_chr_offsets();
    }
}
