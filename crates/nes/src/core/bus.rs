use super::apu::Apu;
use super::common::{NesAddr, NesByte};
use super::joypad::{Joypad, NUM_PLAYERS, PLAYER1, PLAYER2};
use super::log::*;
use super::mappers::MapperKind;
use super::ppu::*;

// CPU memory map constants
// ref: https://www.nesdev.org/wiki/CPU_memory_map
const CPU_RAM_SIZE: usize = 0x800; // 2KB internal RAM
const CPU_RAM_MASK: NesAddr = 0x07FF; // RAM mirroring mask
const PPU_REG_MASK: NesAddr = 0x0007; // PPU register mirroring mask
const EXTENDED_RAM_START: NesAddr = 0x6000;
const EXTENDED_RAM_SIZE: usize = 0x2000; // 8KB

// I/O registers
// ref: https://www.nesdev.org/wiki/2A03
const OAMDMA: NesAddr = 0x4014;
const JOY1: NesAddr = 0x4016;
const JOY2: NesAddr = 0x4017;

// PPU register: https://www.nesdev.org/wiki/PPU_registers
pub trait Bus {
    fn read(&mut self, addr: NesAddr) -> NesByte;
    fn write(&mut self, addr: NesAddr, value: NesByte);
}

/// OAM DMA stalls the CPU for 513 or 514 cycles
/// ref: https://www.nesdev.org/wiki/PPU_OAM#DMA
const OAM_DMA_CYCLES: u32 = 513;

pub struct CpuBus {
    /// Internal RAM (2KB, mirrored to $0000-$1FFF)
    ram: [NesByte; CPU_RAM_SIZE],
    /// Extended RAM at $6000-$7FFF (if mapper supports it)
    extended_ram: Vec<NesByte>,
    /// Cached at mapper load time (mapper lives on `PpuBus` inside `ppu`;
    /// this flag avoids reaching through it on every extended-RAM access).
    has_extended_ram: bool,
    /// The PPU, owned directly (no `Rc<RefCell>`): the CPU is the only reader
    /// of PPU state outside of `Emulator`, and `Emulator` reaches it through
    /// `ppu_mut()`. `Box` keeps the ~240KB screen buffer off the stack.
    /// The cartridge mapper lives on `ppu.bus` to keep it in exactly one
    /// place (PRG reads route through it too — see `read_mapper_prg_rom`).
    ppu: Box<Ppu>,
    /// Player controllers
    joypads: [Joypad; NUM_PLAYERS],
    /// Audio Processing Unit
    pub apu: Apu,
    /// Pending DMA cycles to stall the CPU
    pub dma_cycles: u32,
}

// PPU memory map: https://www.nesdev.org/wiki/PPU_memory_map
//
//  +-----------------------------------+
//  | $0000-$1FFF | Pattern tables      |  CHR ROM/RAM (via mapper)
//  +-----------------------------------+
//  | $2000-$2FFF | Nametables          |  VRAM (2KB with mirroring)
//  +-----------------------------------+
//  | $3000-$3EFF | Nametable mirrors   |  mirror of $2000-$2EFF
//  +-----------------------------------+
//  | $3F00-$3F1F | Palettes            |  palette RAM (32 bytes)
//  +-----------------------------------+
//  | $3F20-$3FFF | Palette mirrors     |  mirror of $3F00-$3F1F
//  +-----------------------------------+

// PPU address space boundaries
const PATTERN_TABLE_END: NesAddr = 0x2000;
const NAMETABLE_START: NesAddr = 0x2000;
const PALETTE_START: NesAddr = 0x3F00;
const PALETTE_END: NesAddr = 0x4000;

// Nametable constants
const NT_RAM_SIZE: usize = 0x800; // 2KB VRAM
const NT_SIZE: usize = 0x400; // 1KB per nametable
const NT_MASK: NesAddr = 0x3FF; // 1KB per nametable
const NT_INDEX_MASK: usize = 3; // 4 nametables (0-3)
const PALETTE_SIZE: usize = 0x20; // 32 bytes
const PALETTE_MASK: NesAddr = 0x1F;

/// Normalize palette RAM index: $3F10/$3F14/$3F18/$3F1C mirror $3F00/$3F04/$3F08/$3F0C
/// ref: https://www.nesdev.org/wiki/PPU_palettes#Memory_Map
#[inline]
fn palette_mirror_index(index: usize) -> usize {
    if index >= 0x10 && index & 0x03 == 0 {
        index & 0x0F
    } else {
        index
    }
}

// Mirroring modes
// ref: https://www.nesdev.org/wiki/Mirroring#Nametable_Mirroring
const MIRROR_HORIZONTAL: u8 = 0;
const MIRROR_VERTICAL: u8 = 1;
const MIRROR_ONE_SCREEN_LOWER: u8 = 2;
const MIRROR_ONE_SCREEN_UPPER: u8 = 3;

pub struct PpuBus {
    /// VRAM for nametables (2KB)
    ram: [NesByte; NT_RAM_SIZE],
    /// Nametable offsets into RAM (set by mirroring mode)
    name_tables: [usize; 4],
    /// Palette RAM (32 bytes)
    palette: [NesByte; PALETTE_SIZE],
    /// The cartridge mapper. Owned here rather than shared: `CpuBus` routes
    /// its PRG-space reads/writes through `self.ppu.bus.mapper` so that only
    /// one copy exists and no `Rc<RefCell>` indirection is needed.
    pub(super) mapper: Option<MapperKind>,
}

// CPU memory map: https://www.nesdev.org/wiki/CPU_memory_map
//
//  +-----------------------------------+
//  | $0000-$07FF | Internal RAM (2KB)  |  mirrored at $0800-$1FFF
//  +-----------------------------------+
//  | $2000-$2007 | PPU registers       |  mirrored at $2008-$3FFF
//  +-----------------------------------+
//  | $4000-$4017 | APU & I/O registers |
//  +-----------------------------------+
//  | $4020-$5FFF | Expansion ROM       |  cartridge space (rarely used)
//  +-----------------------------------+
//  | $6000-$7FFF | Extended RAM (8KB)  |  battery-backed save RAM
//  +-----------------------------------+
//  | $8000-$FFFF | PRG ROM (32KB)      |  mapped by cartridge mapper
//  +-----------------------------------+
impl Bus for CpuBus {
    #[inline]
    fn read(&mut self, addr: NesAddr) -> NesByte {
        match addr {
            0x0000..0x2000 => self.read_cpu_ram(addr),
            0x2000..0x4000 => self.read_ppu_register(addr),
            0x4000..0x4018 => self.read_io_register(addr),
            0x4020..0x6000 => self.read_expansion_rom(addr),
            0x6000..0x8000 => self.read_extended_ram(addr),
            0x8000..=0xFFFF => self.read_mapper_prg_rom(addr),
            _ => 0,
        }
    }

    fn write(&mut self, addr: NesAddr, value: NesByte) {
        match addr {
            0x0000..0x2000 => self.write_cpu_ram(addr, value),
            0x2000..0x4000 => self.write_ppu_register(addr, value),
            0x4000..0x4018 => self.write_io_register(addr, value),
            0x4020..0x6000 => self.write_expansion_rom(addr, value),
            0x6000..0x8000 => self.write_extended_ram(addr, value),
            0x8000..=0xFFFF => self.write_mapper_prg_rom(addr, value),
            _ => {}
        }
    }
}

impl CpuBus {
    /// Access the 2KB internal RAM.
    pub fn ram(&self) -> &[u8] {
        &self.ram
    }

    /// Mutable access to the 2KB internal RAM.
    pub fn ram_mut(&mut self) -> &mut [u8] {
        &mut self.ram
    }

    pub fn new(ppu: Ppu, joypads: [Joypad; NUM_PLAYERS]) -> Self {
        Self {
            ram: [0; CPU_RAM_SIZE],
            extended_ram: Vec::new(),
            has_extended_ram: false,
            ppu: Box::new(ppu),
            joypads,
            apu: Apu::new(),
            dma_cycles: 0,
        }
    }

    /// Install the cartridge mapper. Stores it on `PpuBus`; `CpuBus` routes
    /// PRG-space access through `self.ppu.bus.mapper` on demand. Also sizes
    /// extended RAM and caches the `has_extended_ram` flag.
    pub fn set_mapper(&mut self, mapper: MapperKind) {
        self.has_extended_ram = mapper.has_extended_ram();
        if self.has_extended_ram {
            self.extended_ram.resize(EXTENDED_RAM_SIZE, 0);
        }
        self.ppu.bus.set_mapper(mapper);
    }

    /// Shared access to the mapper (if loaded).
    #[inline]
    pub fn mapper(&self) -> Option<&MapperKind> {
        self.ppu.bus.mapper.as_ref()
    }

    /// Exclusive access to the mapper (if loaded).
    #[inline]
    pub fn mapper_mut(&mut self) -> Option<&mut MapperKind> {
        self.ppu.bus.mapper.as_mut()
    }

    /// Shared access to the PPU.
    #[inline]
    pub fn ppu(&self) -> &Ppu {
        &self.ppu
    }

    /// Exclusive access to the PPU.
    #[inline]
    pub fn ppu_mut(&mut self) -> &mut Ppu {
        &mut self.ppu
    }

    /// Set joypad button state for a player
    pub fn set_joypad(&mut self, player: usize, state: NesByte) {
        self.joypads[player].set_buttons(state);
    }

    /// Return a pointer to a 256-byte page in memory (for OAM DMA)
    /// ref: https://www.nesdev.org/wiki/PPU_OAM#DMA
    fn get_page(&self, page: NesByte) -> Option<&[NesByte]> {
        let addr = (page as NesAddr) << 8;
        match addr {
            0x0000..0x2000 => self.get_ram_page(addr),
            0x6000..0x8000 => self.get_extended_ram_page(addr),
            _ => None, // other ranges handled by slow path in write_oam_dma
        }
    }

    /// Get a 256-byte page from internal RAM ($0000-$1FFF)
    fn get_ram_page(&self, addr: NesAddr) -> Option<&[NesByte]> {
        let start = (addr & CPU_RAM_MASK) as usize;
        Some(&self.ram[start..start + 256])
    }

    /// Get a 256-byte page from extended (PRG) RAM ($6000-$7FFF)
    fn get_extended_ram_page(&self, addr: NesAddr) -> Option<&[NesByte]> {
        if !self.extended_ram.is_empty() {
            let start = (addr - EXTENDED_RAM_START) as usize;
            Some(&self.extended_ram[start..start + 256])
        } else {
            None
        }
    }

    /// Read from internal RAM ($0000-$1FFF, mirrored every $800)
    fn read_cpu_ram(&self, addr: NesAddr) -> NesByte {
        self.ram[(addr & CPU_RAM_MASK) as usize]
    }

    /// Read PPU register ($2000-$3FFF, mirrored every 8 bytes)
    /// ref: https://www.nesdev.org/wiki/PPU_registers
    fn read_ppu_register(&mut self, addr: NesAddr) -> NesByte {
        let reg = (addr & PPU_REG_MASK) | 0x2000;
        match reg {
            PPU_PPUSTATUS => self.ppu.status(),
            PPU_OAMDATA => self.ppu.oamdata(),
            PPU_PPUDATA => self.ppu.get_data(),
            _ => 0,
        }
    }

    /// Read I/O register ($4000-$4017)
    /// ref: https://www.nesdev.org/wiki/2A03
    fn read_io_register(&mut self, addr: NesAddr) -> NesByte {
        match addr {
            0x4015 => self.apu.read_status(),
            JOY1 => self.joypads[PLAYER1].read(),
            JOY2 => self.joypads[PLAYER2].read(),
            _ => 0,
        }
    }

    /// Read from expansion ROM ($4020-$5FFF)
    fn read_expansion_rom(&self, addr: NesAddr) -> NesByte {
        nes_warn!("Expansion ROM read at: {:#06X}", addr);
        0
    }

    /// Read from extended RAM ($6000-$7FFF)
    #[inline]
    fn read_extended_ram(&self, addr: NesAddr) -> NesByte {
        if self.has_extended_ram {
            return self.extended_ram[(addr - EXTENDED_RAM_START) as usize];
        }
        0
    }

    /// Read from PRG ROM via mapper ($8000-$FFFF)
    #[inline]
    fn read_mapper_prg_rom(&self, addr: NesAddr) -> NesByte {
        match &self.ppu.bus.mapper {
            Some(m) => m.read_prg(addr),
            None => 0,
        }
    }

    /// Write to internal RAM ($0000-$1FFF)
    fn write_cpu_ram(&mut self, addr: NesAddr, value: NesByte) {
        self.ram[(addr & CPU_RAM_MASK) as usize] = value;
    }

    /// Write to PPU register ($2000-$3FFF)
    /// ref: https://www.nesdev.org/wiki/PPU_registers
    fn write_ppu_register(&mut self, addr: NesAddr, value: NesByte) {
        let reg = (addr & PPU_REG_MASK) | 0x2000;
        match reg {
            PPU_PPUCTRL => self.ppu.control(value),
            PPU_PPUMASK => self.ppu.set_mask(value),
            PPU_OAMADDR => self.ppu.set_oam_addr(value),
            PPU_OAMDATA => self.ppu.set_oam_data(value),
            PPU_PPUSCROL => self.ppu.set_scroll(value),
            PPU_PPUADDR => self.ppu.set_data_address(value),
            PPU_PPUDATA => self.ppu.set_data(value),
            _ => {}
        }
    }

    /// Write to I/O register ($4000-$4017)
    /// ref: https://www.nesdev.org/wiki/2A03
    fn write_io_register(&mut self, addr: NesAddr, value: NesByte) {
        match addr {
            0x4000..0x4014 | 0x4015 | 0x4017 => self.apu.write(addr, value),
            OAMDMA => self.write_oam_dma(value),
            JOY1 => self.write_joypad_strobe(value),
            _ => {}
        }
    }

    /// OAM DMA: copy 256 bytes from CPU page to PPU OAM
    /// Supports all CPU address ranges (RAM, extended RAM, and PRG ROM).
    /// ref: https://www.nesdev.org/wiki/PPU_OAM#DMA
    fn write_oam_dma(&mut self, page: NesByte) {
        // Stage the page into a local buffer first: borrowing from `self.ram`
        // or `self.extended_ram` for the fast path can't coexist with the
        // `&mut self.ppu` needed below (disjoint-field borrow splitting
        // doesn't work through method calls), and the 256-byte copy is
        // negligible compared to the DMA's 513-cycle stall.
        let mut buf = [0u8; 256];
        if let Some(data) = self.get_page(page) {
            buf.copy_from_slice(data);
        } else {
            // Slow path: read each byte via the bus (handles PRG ROM, etc.)
            let base = (page as NesAddr) << 8;
            for (i, byte) in buf.iter_mut().enumerate() {
                *byte = self.read(base + i as u16);
            }
        }
        for &byte in &buf {
            self.ppu.set_oam_data(byte);
        }
        self.dma_cycles = OAM_DMA_CYCLES;
    }

    /// Joypad strobe: writing 1 then 0 latches button state
    /// ref: https://www.nesdev.org/wiki/Standard_controller
    fn write_joypad_strobe(&mut self, value: NesByte) {
        for joypad in &mut self.joypads {
            joypad.write_strobe(value);
        }
    }

    /// Write to expansion ROM ($4020-$5FFF)
    fn write_expansion_rom(&mut self, addr: NesAddr, value: NesByte) {
        nes_warn!("Expansion ROM write at: {:#06X}", addr);
    }

    /// Write to extended RAM ($6000-$7FFF)
    #[inline]
    fn write_extended_ram(&mut self, addr: NesAddr, value: NesByte) {
        if self.has_extended_ram {
            self.extended_ram[(addr - EXTENDED_RAM_START) as usize] = value;
        }
    }

    /// Write to PRG ROM via mapper ($8000-$FFFF)
    #[inline]
    fn write_mapper_prg_rom(&mut self, addr: NesAddr, value: NesByte) {
        if let Some(ref mut mapper) = self.ppu.bus.mapper {
            mapper.write_prg(addr, value);
        }
    }
}

impl Default for PpuBus {
    fn default() -> Self {
        Self::new()
    }
}
impl PpuBus {
    pub fn new() -> Self {
        Self {
            ram: [0; NT_RAM_SIZE],
            name_tables: [0; 4],
            palette: [0; PALETTE_SIZE],
            mapper: None,
        }
    }

    pub fn set_mapper(&mut self, mapper: MapperKind) {
        let mirroring = mapper.cartridge().name_table_mirroring();
        self.mapper = Some(mapper);
        self.update_mirroring(mirroring);
    }

    /// Clear nametable RAM and palette RAM. Called from `Ppu::reset` so
    /// stale tiles from a prior episode can't leak into a fresh observation.
    pub fn reset(&mut self) {
        self.ram.fill(0);
        self.palette.fill(0);
    }

    /// Read a color index from palette RAM
    /// ref: https://www.nesdev.org/wiki/PPU_palettes#Memory_Map
    #[inline]
    pub fn read_palette(&self, addr: NesByte) -> NesByte {
        self.palette[palette_mirror_index((addr & PALETTE_MASK as u8) as usize)]
    }

    /// Update nametable mirroring offsets
    /// ref: https://www.nesdev.org/wiki/Mirroring#Nametable_Mirroring
    pub fn update_mirroring(&mut self, mirroring: u8) {
        match mirroring {
            MIRROR_HORIZONTAL => self.name_tables = [0, 0, NT_SIZE, NT_SIZE],
            MIRROR_VERTICAL => self.name_tables = [0, NT_SIZE, 0, NT_SIZE],
            MIRROR_ONE_SCREEN_LOWER => self.name_tables = [0, 0, 0, 0],
            MIRROR_ONE_SCREEN_UPPER => self.name_tables = [NT_SIZE, NT_SIZE, NT_SIZE, NT_SIZE],
            _ => {
                nes_warn!("Unsupported nametable mirroring mode: {}", mirroring);
                self.name_tables = [0, 0, 0, 0];
            }
        }
    }

    /// Read from CHR ROM/RAM via mapper ($0000-$1FFF)
    #[inline]
    fn read_chr(&self, addr: NesAddr) -> NesByte {
        match &self.mapper {
            Some(m) => m.read_chr(addr),
            None => 0,
        }
    }

    /// Write to CHR ROM/RAM via mapper ($0000-$1FFF)
    #[inline]
    fn write_chr(&mut self, addr: NesAddr, value: NesByte) {
        if let Some(ref mut mapper) = self.mapper {
            mapper.write_chr(addr, value);
        }
    }

    /// Compute the flat RAM offset for a nametable address.
    #[inline]
    fn nametable_offset(&self, addr: NesAddr) -> usize {
        let nt_index = ((addr - NAMETABLE_START) / NT_SIZE as u16) as usize & NT_INDEX_MASK;
        self.name_tables[nt_index] + (addr & NT_MASK) as usize
    }

    /// Read from nametable VRAM ($2000-$3EFF)
    fn read_nametable(&self, addr: NesAddr) -> NesByte {
        self.ram[self.nametable_offset(addr)]
    }

    /// Write to nametable VRAM ($2000-$3EFF)
    fn write_nametable(&mut self, addr: NesAddr, value: NesByte) {
        self.ram[self.nametable_offset(addr)] = value;
    }

    /// Read from palette RAM ($3F00-$3FFF)
    fn read_palette_ram(&self, addr: NesAddr) -> NesByte {
        self.palette[palette_mirror_index((addr & PALETTE_MASK) as usize)]
    }

    /// Write to palette RAM ($3F00-$3FFF)
    /// ref: https://www.nesdev.org/wiki/PPU_palettes#Memory_Map
    fn write_palette_ram(&mut self, addr: NesAddr, value: NesByte) {
        self.palette[palette_mirror_index((addr & PALETTE_MASK) as usize)] = value;
    }
}

impl Bus for PpuBus {
    #[inline]
    fn read(&mut self, addr: NesAddr) -> NesByte {
        match addr {
            0x0000..PATTERN_TABLE_END => self.read_chr(addr),
            NAMETABLE_START..PALETTE_START => self.read_nametable(addr),
            PALETTE_START..PALETTE_END => self.read_palette_ram(addr),
            _ => 0,
        }
    }

    fn write(&mut self, addr: NesAddr, value: NesByte) {
        match addr {
            0x0000..PATTERN_TABLE_END => self.write_chr(addr, value),
            NAMETABLE_START..PALETTE_START => self.write_nametable(addr, value),
            PALETTE_START..PALETTE_END => self.write_palette_ram(addr, value),
            _ => {}
        }
    }
}
