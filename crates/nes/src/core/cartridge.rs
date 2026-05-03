use super::common::NesByte;
use std::io;

// iNES header constants
const INES_HEADER_SIZE: usize = 0x10;
const INES_MAGIC: &[u8; 4] = b"NES\x1A";
const PRG_ROM_BANK_SIZE: usize = 0x4000; // 16KB per PRG ROM bank
const CHR_ROM_BANK_SIZE: usize = 0x2000; // 8KB per CHR ROM bank

// Flags 6 bitmasks
const FLAGS6_MIRROR: u8 = 0x09; // bits 0,3: mirroring + four-screen
const FLAGS6_BATTERY: u8 = 0x02; // bit 1: battery-backed RAM
const FLAGS6_TRAINER: u8 = 0x04; // bit 2: 512-byte trainer present
const FLAGS6_MAPPER_LO: u8 = 0x0F; // bits 4-7 (after >> 4): mapper lower nibble

// Flags 7 bitmasks
const FLAGS7_MAPPER_HI: u8 = 0xF0; // bits 4-7: mapper upper nibble

// iNES header byte offsets
const INES_PRG_ROM_SIZE: usize = 4; // header byte 4: PRG ROM bank count
const INES_CHR_ROM_SIZE: usize = 5; // header byte 5: CHR ROM bank count
const INES_FLAGS6: usize = 6; // header byte 6: flags 6
const INES_FLAGS7: usize = 7; // header byte 7: flags 7
const TRAINER_SIZE: usize = 512; // 512-byte trainer

pub struct Cartridge {
    /// The PRG ROM (program code)
    prg_rom: Vec<NesByte>,

    /// The CHR ROM (pattern/tile data)
    chr_rom: Vec<NesByte>,

    /// The name table mirroring mode
    /// ref: https://www.nesdev.org/wiki/Mirroring#Nametable_Mirroring
    name_table_mirroring: NesByte,

    /// The mapper ID number
    /// ref: https://www.nesdev.org/wiki/Mapper
    mapper_number: NesByte,

    /// Whether this cartridge uses extended RAM
    has_extended_ram: bool,
}

// iNES Header (16 bytes) — ref: https://www.nesdev.org/wiki/INES
//   0                   1                   2                   3
//   0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1 2 3 4 5 6 7 8 9 0 1
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |      'N'      |      'E'      |      'S'      |     0x1A      |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |  PRG ROM Size |  CHR ROM Size |    Flags 6    |    Flags 7    |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |                         Padding (unused)                      |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//  |                         Padding (unused)                      |
//  +-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+-+
//
//  Flags 6 (byte 6):
//
//   7   6   5   4   3   2   1   0
//  +---+---+---+---+---+---+---+---+
//  | N | N | N | N | F | T | B | M |
//  +---+---+---+---+---+---+---+---+
//    |   |   |   |   |   |   |   |
//    |   |   |   |   |   |   |   +-- Mirroring (0=Horizontal, 1=Vertical)
//    |   |   |   |   |   |   +------ Battery-backed RAM ($6000-$7FFF)
//    |   |   |   |   |   +---------- Trainer (512B at $7000-$71FF)
//    |   |   |   |   +-------------- Four-screen VRAM
//    +---+---+---+------------------ Mapper number (lower nibble)
//
//  Flags 7 (byte 7):
//
//   7   6   5   4   3   2   1   0
//  +---+---+---+---+---+---+---+---+
//  | N | N | N | N | x | x | x | x |
//  +---+---+---+---+---+---+---+---+
//    |   |   |   |
//    +---+---+---+------------------ Mapper number (upper nibble)
//
//  Full file layout:
//
//  +-----------------------------------+
//  |  Header (16 bytes)                |  0x00 - 0x0F
//  +-----------------------------------+
//  |  Trainer (512 bytes, if present)  |  optional, if Flags6 bit 2 set
//  +-----------------------------------+
//  |  PRG ROM (16KB x N banks)         |  header[4] = N
//  +-----------------------------------+
//  |  CHR ROM (8KB x N banks)          |  header[5] = N (0 = uses CHR RAM)
//  +-----------------------------------+

impl Default for Cartridge {
    fn default() -> Self {
        Self::new()
    }
}
impl Cartridge {
    pub fn new() -> Self {
        Self {
            prg_rom: Vec::new(),
            chr_rom: Vec::new(),
            name_table_mirroring: 0,
            mapper_number: 0,
            has_extended_ram: false,
        }
    }

    pub fn prg_rom(&self) -> &[NesByte] {
        &self.prg_rom
    }

    pub fn chr_rom(&self) -> &[NesByte] {
        &self.chr_rom
    }

    pub fn mapper_number(&self) -> NesByte {
        self.mapper_number
    }

    pub fn name_table_mirroring(&self) -> NesByte {
        self.name_table_mirroring
    }

    pub fn has_extended_ram(&self) -> bool {
        self.has_extended_ram
    }

    pub fn load(&mut self, path: &str) -> io::Result<()> {
        let data = std::fs::read(path)?;
        self.load_bytes(&data)
    }

    pub fn load_bytes(&mut self, data: &[u8]) -> io::Result<()> {
        if data.len() < INES_HEADER_SIZE {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "ROM file too small",
            ));
        }
        if &data[..4] != INES_MAGIC {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "not an iNES ROM (missing 'NES\\x1A' magic)",
            ));
        }
        let header = &data[..INES_HEADER_SIZE];
        self.read_internal_data(header);

        // ref: https://www.nesdev.org/wiki/CHR_ROM_vs._CHR_RAM
        let prg_end = self.read_prg_rom(header, data)?;
        self.read_chr_rom(header, data, prg_end)?;
        Ok(())
    }

    fn read_internal_data(&mut self, header: &[NesByte]) {
        self.name_table_mirroring = header[INES_FLAGS6] & FLAGS6_MIRROR;
        let mapper_lo = (header[INES_FLAGS6] >> 4) & FLAGS6_MAPPER_LO;
        let mapper_hi = header[INES_FLAGS7] & FLAGS7_MAPPER_HI;
        self.mapper_number = mapper_lo | mapper_hi;
        self.has_extended_ram = header[INES_FLAGS6] & FLAGS6_BATTERY != 0;
    }

    fn read_prg_rom(&mut self, header: &[NesByte], data: &[NesByte]) -> io::Result<usize> {
        let banks = header[INES_PRG_ROM_SIZE] as usize;
        if banks == 0 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "iNES header declares zero PRG ROM banks",
            ));
        }
        let has_trainer = header[INES_FLAGS6] & FLAGS6_TRAINER != 0;
        let prg_start = INES_HEADER_SIZE + if has_trainer { TRAINER_SIZE } else { 0 };
        let prg_end = prg_start + PRG_ROM_BANK_SIZE * banks;
        if prg_end > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "PRG ROM data truncated",
            ));
        }
        self.prg_rom = data[prg_start..prg_end].to_vec();
        Ok(prg_end)
    }

    fn read_chr_rom(
        &mut self,
        header: &[NesByte],
        data: &[NesByte],
        prg_end: usize,
    ) -> io::Result<()> {
        let vbanks = header[INES_CHR_ROM_SIZE] as usize;
        if vbanks == 0 {
            return Ok(());
        }
        let chr_end = prg_end + CHR_ROM_BANK_SIZE * vbanks;
        if chr_end > data.len() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "CHR ROM data truncated",
            ));
        }
        self.chr_rom = data[prg_end..chr_end].to_vec();
        Ok(())
    }
}
