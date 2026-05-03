use super::bus::{Bus, PpuBus};
use super::common::{NesAddr, NesByte, NesPixel};
use super::palette::PALETTE;

// PPU reference:
//
// PPU overview: https://www.nesdev.org/wiki/PPU
// PPU registers: https://www.nesdev.org/wiki/PPU_registers
// PPU rendering: https://www.nesdev.org/wiki/PPU_rendering
// PPU OAM (sprites): https://www.nesdev.org/wiki/PPU_OAM
// PPU nametables: https://www.nesdev.org/wiki/PPU_nametables
// PPU palettes: https://www.nesdev.org/wiki/PPU_palettes

// PPU register addresses
// ref: https://www.nesdev.org/wiki/PPU_registers
pub const PPU_PPUCTRL: u16 = 0x2000;
pub const PPU_PPUMASK: u16 = 0x2001;
pub const PPU_PPUSTATUS: u16 = 0x2002;
pub const PPU_OAMADDR: u16 = 0x2003;
pub const PPU_OAMDATA: u16 = 0x2004;
pub const PPU_PPUSCROL: u16 = 0x2005;
pub const PPU_PPUADDR: u16 = 0x2006;
pub const PPU_PPUDATA: u16 = 0x2007;
pub const PPU_OAMDMA: u16 = 0x4014;

// Timing constants
const VISIBLE_SCANLINES: i32 = 240;
const SCANLINE_VISIBLE_DOTS: i32 = 256;
const SCANLINE_END_CYCLE: i32 = 341;
const SPRITE_FETCH_CYCLE: i32 = 260; // A12 rising edge during sprite tile fetches
const VERTICAL_COPY_START: i32 = 280; // start of vertical scroll copy range (pre-render)
const VERTICAL_COPY_END: i32 = 304; // end of vertical scroll copy range (pre-render)
const FRAME_END_SCANLINE: i32 = 261;

// OAM constants
const OAM_SIZE: usize = 256; // 64 sprites x 4 bytes
const OAM_ENTRY_SIZE: usize = 4; // bytes per sprite: Y, tile, attribute, X
const NUM_SPRITES: usize = 64;
const MAX_SPRITES_PER_SCANLINE: usize = 8;
const SPRITE_HEIGHT_NORMAL: i32 = 8; // 8x8 sprites
const SPRITE_HEIGHT_TALL: i32 = 16; // 8x16 sprites

// Sprite attribute bit masks
const SPRITE_FLIP_HORIZONTAL: NesByte = 0x40;
const SPRITE_FLIP_VERTICAL: NesByte = 0x80;
const SPRITE_PRIORITY: NesByte = 0x20;
const SPRITE_PALETTE_MASK: NesByte = 0x03;

// PPUCTRL bit masks
const CTRL_NMI_ENABLE: NesByte = 0x80;
const CTRL_LONG_SPRITES: NesByte = 0x20;
const CTRL_BG_PAGE: NesByte = 0x10;
const CTRL_SPRITE_PAGE: NesByte = 0x08;
const CTRL_ADDR_INCREMENT: NesByte = 0x04;
const CTRL_NAMETABLE_MASK: NesByte = 0x03;

// PPUMASK bit masks
const MASK_SHOW_BG_LEFT: NesByte = 0x02;
const MASK_SHOW_SPR_LEFT: NesByte = 0x04;
const MASK_SHOW_BG: NesByte = 0x08;
const MASK_SHOW_SPR: NesByte = 0x10;

// PPUSTATUS bit masks
const STATUS_SPRITE_OVERFLOW: NesByte = 0x20;
const STATUS_SPRITE_ZERO_HIT: NesByte = 0x40;
const STATUS_VBLANK: NesByte = 0x80;

// Character page select
const CHR_PAGE_LOW: NesAddr = 0;
const CHR_PAGE_HIGH: NesAddr = 1;

// Loopy register bit fields (data_address / temp_address)
// ref: https://www.nesdev.org/wiki/PPU_scrolling#PPU_internal_registers
//
//   yyy NN YYYYY XXXXX
//   ||| || ||||| +++++-- coarse X scroll
//   ||| || +++++-------- coarse Y scroll
//   ||| ++-------------- nametable select
//   +++----------------- fine Y scroll
const LOOPY_COARSE_X: NesAddr = 0x001F;
const LOOPY_COARSE_Y: NesAddr = 0x03E0;
const LOOPY_NAMETABLE: NesAddr = 0x0C00;
const LOOPY_NAMETABLE_H: NesAddr = 0x0400;
const LOOPY_NAMETABLE_V: NesAddr = 0x0800;
const LOOPY_FINE_Y: NesAddr = 0x7000;
const LOOPY_FINE_Y_INC: NesAddr = 0x1000; // increment step for fine Y (1 << 12)
/// Horizontal bits: coarse X + horizontal nametable
const LOOPY_HORIZONTAL: NesAddr = LOOPY_COARSE_X | LOOPY_NAMETABLE_H; // 0x041F
/// Vertical bits: coarse Y + fine Y + vertical nametable
const LOOPY_VERTICAL: NesAddr = LOOPY_COARSE_Y | LOOPY_FINE_Y | LOOPY_NAMETABLE_V; // 0x7BE0
/// Scroll bits set by first PPUSCROLL write (coarse X)
const LOOPY_SCROLL_X: NesAddr = LOOPY_COARSE_X; // 0x001F
/// Scroll bits set by second PPUSCROLL write (coarse Y + fine Y)
const LOOPY_SCROLL_Y: NesAddr = LOOPY_COARSE_Y | LOOPY_FINE_Y; // 0x73E0
/// High byte mask for PPUADDR first write
const LOOPY_HIGH_BYTE: NesAddr = 0xFF00;
/// Low byte mask for PPUADDR second write
const LOOPY_LOW_BYTE: NesAddr = 0x00FF;
/// PPUADDR high byte valid bits (bit 14 cleared)
const PPUADDR_HIGH_MASK: NesAddr = 0x3F;
/// PPU address space is 14 bits ($0000-$3FFF)
const PPU_ADDR_MASK: NesAddr = 0x3FFF;

// Scroll byte bit masks
// ref: https://www.nesdev.org/wiki/PPU_scrolling#Register_controls
/// Fine scroll (low 3 bits of scroll byte)
const SCROLL_FINE_MASK: NesByte = 0x07;
/// Coarse scroll (high 5 bits of scroll byte)
const SCROLL_COARSE_MASK: NesByte = 0xF8;
/// Fine Y is stored at bits 12-14 of temp_address
const FINE_Y_SHIFT: u8 = 12;
/// Coarse Y is stored at bits 5-9 of temp_address
const COARSE_Y_SHIFT: u8 = 2;

// NES nametable tile grid dimensions (32x30 visible tiles)
// ref: https://www.nesdev.org/wiki/PPU_nametables
const COARSE_X_MAX: NesAddr = 31; // last coarse X tile (0-31, 32 tiles per row)
const COARSE_Y_LAST_ROW: NesAddr = 29; // last visible tile row (0-29, 30 rows)
const COARSE_Y_MAX: NesAddr = 31; // attribute table overflow row (wraps without nametable switch)

// Nametable address constants
const NAMETABLE_BASE: NesAddr = 0x2000;
const NAMETABLE_ADDR_MASK: NesAddr = 0x0FFF;
const ATTR_TABLE_BASE: NesAddr = 0x23C0;
// Attribute table address bit masks (for computing attr byte position)
const ATTR_BLOCK_MASK: NesAddr = 0x38; // coarse Y / 4
const ATTR_COLUMN_MASK: NesAddr = 0x07; // coarse X / 4
const ATTR_QUADRANT_Y_BIT: NesAddr = 4; // bit set if in bottom half of attr block
const ATTR_QUADRANT_X_BIT: NesAddr = 2; // bit set if in right half of attr block

// OAM field offsets within each 4-byte sprite entry
const OAM_Y: usize = 0;
const OAM_TILE: usize = 1;
const OAM_ATTR: usize = 2;
const OAM_X: usize = 3;

// Palette address space boundary
const PALETTE_ADDR_START: NesAddr = 0x3F00;

// CHR page size
const CHR_PAGE_OFFSET: NesAddr = 0x1000;

// Pattern table / tile constants
const FINE_Y_MASK: NesAddr = 0x07;
const TILE_BYTE_SIZE: NesAddr = 16; // bytes per 8x8 tile in pattern table
const CHR_PLANE_OFFSET: NesAddr = 8; // offset between low and high bitplane
const TILE_PIXELS: i32 = 8; // pixels per tile side
const PALETTE_BITS_MASK: NesByte = 0x03;
const SPRITE_PALETTE_BASE: NesByte = 0x10; // sprite palettes start at $10 in palette RAM

/// Advance a PPU address by `increment`, wrapping to 14-bit address space.
#[inline]
fn advance_address(addr: NesAddr, increment: NesAddr) -> NesAddr {
    addr.wrapping_add(increment) & PPU_ADDR_MASK
}

/// Replace selected bits in `dst` with corresponding bits from `src`.
#[inline]
fn merge_bits(dst: NesAddr, src: NesAddr, mask: NesAddr) -> NesAddr {
    (dst & !mask) | (src & mask)
}

/// Pipeline state of the PPU
#[derive(Clone, Copy, PartialEq)]
enum PipelineState {
    PreRender,
    Render,
    PostRender,
    VerticalBlank,
}

pub struct Ppu {
    // Timing
    cycles: i32,
    scanline: i32,
    is_even_frame: bool,
    pipeline_state: PipelineState,

    // Status flags
    is_vblank: bool,
    is_sprite_zero_hit: bool,
    is_sprite_overflow: bool,

    // Internal registers
    /// Current VRAM address (15 bits)
    data_address: NesAddr,
    /// Temporary VRAM address (15 bits)
    temp_address: NesAddr,
    /// Fine X scroll (3 bits)
    fine_x_scroll: NesByte,
    /// Write toggle (first/second write)
    is_first_write: bool,
    /// Read buffer for PPUDATA
    data_buffer: NesByte,
    /// OAM address
    sprite_data_address: NesByte,

    // PPUCTRL decoded fields
    is_long_sprites: bool,
    is_interrupting: bool,
    background_page: NesAddr,
    sprite_page: NesAddr,
    data_address_increment: NesAddr,

    // PPUMASK decoded fields
    is_showing_sprites: bool,
    is_showing_background: bool,
    is_hiding_edge_sprites: bool,
    is_hiding_edge_background: bool,

    // Memory
    /// OAM (sprite) memory — 64 sprites x 4 bytes
    sprite_memory: [NesByte; OAM_SIZE],
    /// Sprites found on the current scanline
    scanline_sprites: Vec<NesByte>,

    // Background tile cache.
    //
    // Each tile is 8 pixels wide; all 8 pixels share the same nametable byte,
    // CHR low/high planes, and attribute byte — only the bit position differs.
    // The original code re-fetched all 4 bytes per pixel.
    //
    // BEFORE (every pixel = 4 bus reads):
    //   ┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐┌──────┐
    //   │4 reads││4 reads││4 reads││4 reads││4 reads││4 reads││4 reads││4 reads│ ...
    //   └──────┘└──────┘└──────┘└──────┘└──────┘└──────┘└──────┘└──────┘
    //   = 4 × 256 × 240 = 245,760 bg reads/frame
    //
    // AFTER (first pixel of tile fetches; rest are bit shifts on cached bytes):
    //   ┌──────┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐┌─┐ ┌──────┐┌─┐┌─┐
    //   │4 reads││▶││▶││▶││▶││▶││▶││▶│ │4 reads││▶││▶│ ...
    //   └──────┘└─┘└─┘└─┘└─┘└─┘└─┘└─┘ └──────┘└─┘└─┘
    //         cached bytes reused 7×        re-fetch
    //   = 4 × 32 × 240 ≈ 30,720 bg reads/frame  (~8× fewer)
    //
    // Correctness invariant: cache is keyed implicitly on (data_address,
    // background_page). Any mutation of those fields MUST clear
    // `bg_cache_valid`. The fetch predicate `x_fine == 0 || !bg_cache_valid`
    // also re-primes the cache on the first visible pixel of every scanline,
    // covering fine_x_scroll > 0 and leftmost-8-px edge-hide.
    bg_tile_lo: NesByte,
    bg_tile_hi: NesByte,
    bg_attr_bits: NesByte,
    bg_cache_valid: bool,

    // Screen buffer (flat: H * W, stack-allocated for cache locality)
    screen: [NesPixel; VISIBLE_SCANLINES as usize * SCANLINE_VISIBLE_DOTS as usize],

    // NMI callback flag (set when vblank NMI should fire)
    pub nmi_pending: bool,
    // Set when a full frame has been rendered (vblank start)
    pub frame_complete: bool,
    // Set when mapper scanline counter should be clocked (A12 rising edge)
    pub scanline_irq: bool,

    /// The PPU bus (pattern tables, nametables, palettes)
    pub bus: PpuBus,
}

// Lifecycle
impl Ppu {
    pub fn new(bus: PpuBus) -> Self {
        Self {
            cycles: 0,
            scanline: 0,
            is_even_frame: true,
            pipeline_state: PipelineState::PreRender,
            is_vblank: false,
            is_sprite_zero_hit: false,
            is_sprite_overflow: false,
            data_address: 0,
            temp_address: 0,
            fine_x_scroll: 0,
            is_first_write: true,
            data_buffer: 0,
            sprite_data_address: 0,
            is_long_sprites: false,
            is_interrupting: false,
            background_page: CHR_PAGE_LOW,
            sprite_page: CHR_PAGE_LOW,
            data_address_increment: 1,
            is_showing_sprites: true,
            is_showing_background: true,
            is_hiding_edge_sprites: false,
            is_hiding_edge_background: false,
            sprite_memory: [0; OAM_SIZE],
            scanline_sprites: Vec::with_capacity(MAX_SPRITES_PER_SCANLINE),
            bg_tile_lo: 0,
            bg_tile_hi: 0,
            bg_attr_bits: 0,
            bg_cache_valid: false,
            screen: [0; VISIBLE_SCANLINES as usize * SCANLINE_VISIBLE_DOTS as usize],
            nmi_pending: false,
            frame_complete: false,
            scanline_irq: false,
            bus,
        }
    }

    pub fn reset(&mut self) {
        self.is_long_sprites = false;
        self.is_interrupting = false;
        self.is_vblank = false;
        self.is_sprite_overflow = false;
        self.is_showing_background = true;
        self.is_showing_sprites = true;
        self.is_even_frame = true;
        self.is_first_write = true;
        self.background_page = CHR_PAGE_LOW;
        self.sprite_page = CHR_PAGE_LOW;
        self.data_address = 0;
        self.cycles = 0;
        self.scanline = 0;
        self.sprite_data_address = 0;
        self.fine_x_scroll = 0;
        self.temp_address = 0;
        self.data_address_increment = 1;
        self.pipeline_state = PipelineState::PreRender;
        self.scanline_sprites.clear();
        self.bg_tile_lo = 0;
        self.bg_tile_hi = 0;
        self.bg_attr_bits = 0;
        self.bg_cache_valid = false;
        self.nmi_pending = false;
        self.frame_complete = false;
        self.scanline_irq = false;
        // Clear OAM + framebuffer so a captured observation during the boot
        // window doesn't leak sprites/pixels from the previous episode.
        self.sprite_memory.fill(0);
        self.screen.fill(0);
        self.bus.reset();
    }

    /// Return the screen buffer as a flat slice — zero allocation.
    pub fn screen_buffer(&self) -> &[NesPixel] {
        &self.screen
    }
}

// Register read/write (mapped to CPU address space)
// ref: https://www.nesdev.org/wiki/PPU_registers
impl Ppu {
    /// PPUCTRL ($2000) — write only
    pub fn control(&mut self, ctrl: NesByte) {
        let was_interrupting = self.is_interrupting;
        self.is_interrupting = ctrl & CTRL_NMI_ENABLE != 0;
        // If NMI was just enabled while vblank is already active, trigger NMI immediately
        // ref: https://www.nesdev.org/wiki/PPU_registers#PPUCTRL
        if !was_interrupting && self.is_interrupting && self.is_vblank {
            self.nmi_pending = true;
        }
        self.is_long_sprites = ctrl & CTRL_LONG_SPRITES != 0;
        let new_background_page = if ctrl & CTRL_BG_PAGE != 0 {
            CHR_PAGE_HIGH
        } else {
            CHR_PAGE_LOW
        };
        if new_background_page != self.background_page {
            // bg page change moves the CHR fetch base address → cached bytes are stale
            self.bg_cache_valid = false;
        }
        self.background_page = new_background_page;
        self.sprite_page = if ctrl & CTRL_SPRITE_PAGE != 0 {
            CHR_PAGE_HIGH
        } else {
            CHR_PAGE_LOW
        };
        self.data_address_increment = if ctrl & CTRL_ADDR_INCREMENT != 0 {
            0x20 // increment by 32 (go down one row)
        } else {
            1 // increment by 1 (go across)
        };
        // Set nametable select bits in temp address
        self.temp_address &= !LOOPY_NAMETABLE;
        self.temp_address |= ((ctrl & CTRL_NAMETABLE_MASK) as NesAddr) << 10;
    }

    /// PPUMASK ($2001) — write only
    pub fn set_mask(&mut self, mask: NesByte) {
        self.is_hiding_edge_background = mask & MASK_SHOW_BG_LEFT == 0;
        self.is_hiding_edge_sprites = mask & MASK_SHOW_SPR_LEFT == 0;
        self.is_showing_background = mask & MASK_SHOW_BG != 0;
        self.is_showing_sprites = mask & MASK_SHOW_SPR != 0;
    }

    /// PPUSTATUS ($2002) — read only
    /// Clears vblank flag and resets write toggle
    pub fn status(&mut self) -> NesByte {
        let mut status = 0;
        if self.is_sprite_overflow {
            status |= STATUS_SPRITE_OVERFLOW;
        }
        if self.is_sprite_zero_hit {
            status |= STATUS_SPRITE_ZERO_HIT;
        }
        if self.is_vblank {
            status |= STATUS_VBLANK;
        }
        self.is_vblank = false;
        self.is_first_write = true;
        status
    }

    /// OAMADDR ($2003) — write only
    pub fn set_oam_addr(&mut self, addr: NesByte) {
        self.sprite_data_address = addr;
    }

    /// OAMDATA ($2004) — read
    pub fn oamdata(&self) -> NesByte {
        self.sprite_memory[self.sprite_data_address as usize]
    }

    /// OAMDATA ($2004) — write
    pub fn set_oam_data(&mut self, value: NesByte) {
        self.sprite_memory[self.sprite_data_address as usize] = value;
        self.sprite_data_address = self.sprite_data_address.wrapping_add(1);
    }

    /// PPUSCROLL ($2005) — write x2
    /// First write: X scroll, second write: Y scroll
    /// ref: https://www.nesdev.org/wiki/PPU_scrolling#$2005_first_write_(w_is_0)
    pub fn set_scroll(&mut self, scroll: NesByte) {
        if self.is_first_write {
            // First write: coarse X = scroll[7:3], fine X = scroll[2:0]
            let coarse_x = (scroll >> 3) as NesAddr;
            self.temp_address &= !LOOPY_SCROLL_X;
            self.temp_address |= coarse_x;
            self.fine_x_scroll = scroll & SCROLL_FINE_MASK;
            self.is_first_write = false;
        } else {
            // Second write: fine Y = scroll[2:0] → t[14:12], coarse Y = scroll[7:3] → t[9:5]
            let fine_y = (scroll as NesAddr & SCROLL_FINE_MASK as NesAddr) << FINE_Y_SHIFT;
            let coarse_y = (scroll as NesAddr & SCROLL_COARSE_MASK as NesAddr) << COARSE_Y_SHIFT;
            self.temp_address &= !LOOPY_SCROLL_Y;
            self.temp_address |= fine_y | coarse_y;
            self.is_first_write = true;
        }
    }

    /// PPUADDR ($2006) — write x2
    /// First write: high byte, second write: low byte
    /// ref: https://www.nesdev.org/wiki/PPU_scrolling#$2006_first_write_(w_is_0)
    pub fn set_data_address(&mut self, address: NesByte) {
        if self.is_first_write {
            let high = (address as NesAddr & PPUADDR_HIGH_MASK) << 8;
            self.temp_address &= !LOOPY_HIGH_BYTE;
            self.temp_address |= high;
            self.is_first_write = false;
        } else {
            self.temp_address &= !LOOPY_LOW_BYTE;
            self.temp_address |= address as NesAddr;
            self.data_address = self.temp_address;
            // data_address mutated → background fetch cache is stale
            self.bg_cache_valid = false;
            self.is_first_write = true;
        }
    }

    /// PPUDATA ($2007) — read
    /// Reads are delayed by one byte when address < $3F00
    /// Palette reads return data immediately but update buffer with nametable data underneath
    /// ref: https://www.nesdev.org/wiki/PPU_registers#The_PPUDATA_read_buffer_(post-fetch)
    pub fn get_data(&mut self) -> NesByte {
        let addr = self.data_address & PPU_ADDR_MASK;
        let mut data = self.bus.read(addr);
        self.advance_data_address();
        // Reads from pattern tables and nametables are buffered
        if addr < PALETTE_ADDR_START {
            std::mem::swap(&mut data, &mut self.data_buffer);
        } else {
            // Palette reads return data directly but put nametable data underneath into buffer
            self.data_buffer = self.bus.read(addr - 0x1000);
        }
        data
    }

    /// PPUDATA ($2007) — write
    pub fn set_data(&mut self, data: NesByte) {
        self.bus.write(self.data_address & PPU_ADDR_MASK, data);
        self.advance_data_address();
    }

    /// OAM DMA — copy 256 bytes into sprite memory, starting at `sprite_data_address`
    /// and wrapping around OAM. The caller must pass exactly `OAM_SIZE` bytes.
    pub fn do_dma(&mut self, page: &[NesByte]) {
        debug_assert_eq!(
            page.len(),
            OAM_SIZE,
            "OAM DMA requires a full OAM-sized page"
        );
        let start = self.sprite_data_address as usize;
        let first_len = OAM_SIZE - start;
        self.sprite_memory[start..].copy_from_slice(&page[..first_len]);
        if start > 0 {
            self.sprite_memory[..start].copy_from_slice(&page[first_len..]);
        }
    }
}

// PPU cycle (rendering pipeline)
// ref: https://www.nesdev.org/wiki/PPU_rendering
impl Ppu {
    /// Execute one PPU cycle. Returns true if mapper scanline counter should be clocked.
    #[inline]
    pub fn cycle(&mut self) -> bool {
        match self.pipeline_state {
            PipelineState::PreRender => self.cycle_pre_render(),
            PipelineState::Render => self.cycle_render(),
            PipelineState::PostRender => self.cycle_post_render(),
            PipelineState::VerticalBlank => self.cycle_vblank(),
        }
        self.cycles += 1;

        if self.scanline_irq {
            self.scanline_irq = false;
            return true;
        }
        false
    }

    /// Whether rendering is enabled (background or sprites are active)
    /// ref: https://www.nesdev.org/wiki/PPU_rendering
    fn is_rendering_enabled(&self) -> bool {
        self.is_showing_background || self.is_showing_sprites
    }

    /// Advance data_address by the configured increment, wrapping to 14-bit PPU address space
    fn advance_data_address(&mut self) {
        self.data_address = advance_address(self.data_address, self.data_address_increment);
        // data_address mutated → background fetch cache is stale
        self.bg_cache_valid = false;
    }

    /// Copy horizontal scroll bits from temp register to current address
    /// ref: https://www.nesdev.org/wiki/PPU_scrolling#At_dot_257_of_each_scanline
    fn copy_horizontal_scroll(&mut self) {
        self.data_address = merge_bits(self.data_address, self.temp_address, LOOPY_HORIZONTAL);
        // data_address mutation invalidates the per-tile background fetch cache
        self.bg_cache_valid = false;
    }

    /// Copy vertical scroll bits from temp register to current address
    /// ref: https://www.nesdev.org/wiki/PPU_scrolling#During_dots_280_to_304_of_the_pre-render_scanline
    fn copy_vertical_scroll(&mut self) {
        self.data_address = merge_bits(self.data_address, self.temp_address, LOOPY_VERTICAL);
        self.bg_cache_valid = false;
    }

    /// Pre-render scanline (scanline -1 / 261)
    /// ref: https://www.nesdev.org/wiki/PPU_rendering#Pre-render_scanline_(-1_or_261)
    fn cycle_pre_render(&mut self) {
        // Cycle 1: clear vblank, sprite zero hit, and sprite overflow flags
        if self.cycles == 1 {
            self.is_vblank = false;
            self.is_sprite_zero_hit = false;
            self.is_sprite_overflow = false;
        }

        let rendering = self.is_rendering_enabled();
        if rendering {
            // Dot 257: reload horizontal scroll from temp register
            // ref: https://www.nesdev.org/wiki/PPU_scrolling#At_dot_257_of_each_scanline
            if self.cycles == SCANLINE_VISIBLE_DOTS + 1 {
                self.copy_horizontal_scroll();
            }

            // Dots 280-304: repeatedly reload vertical scroll from temp register
            // ref: https://www.nesdev.org/wiki/PPU_scrolling#During_dots_280_to_304_of_the_pre-render_scanline
            if self.cycles >= VERTICAL_COPY_START && self.cycles <= VERTICAL_COPY_END {
                self.copy_vertical_scroll();
            }
        }

        // End of scanline: skip last cycle on odd frames when rendering
        let skip = if !self.is_even_frame && rendering {
            1
        } else {
            0
        };
        if self.cycles >= SCANLINE_END_CYCLE - skip {
            self.pipeline_state = PipelineState::Render;
            self.cycles = 0;
            self.scanline = 0;
        }
    }

    /// Visible scanlines (0-239)
    fn cycle_render(&mut self) {
        // Cycles 1-256: render one pixel per cycle
        if self.cycles > 0 && self.cycles <= SCANLINE_VISIBLE_DOTS {
            self.render_pixel();
            // Dot 256: increment vertical scroll
            // ref: https://www.nesdev.org/wiki/PPU_scrolling#At_dot_256_of_each_scanline
            if self.cycles == SCANLINE_VISIBLE_DOTS && self.is_rendering_enabled() {
                self.increment_vertical_scroll();
            }
        } else if self.cycles == SCANLINE_VISIBLE_DOTS + 1 && self.is_rendering_enabled() {
            // Dot 257: copy horizontal scroll bits from t to v
            // ref: https://www.nesdev.org/wiki/PPU_scrolling#At_dot_257_of_each_scanline
            self.copy_horizontal_scroll();
        }

        // A12 rising edge for mapper scanline counter (sprite fetch at cycle 260)
        if self.cycles == SPRITE_FETCH_CYCLE && self.is_rendering_enabled() {
            self.scanline_irq = true;
        }

        if self.cycles >= SCANLINE_END_CYCLE {
            self.evaluate_sprites();
            self.scanline += 1;
            self.cycles = 0;
        }

        if self.scanline >= VISIBLE_SCANLINES {
            self.pipeline_state = PipelineState::PostRender;
        }
    }

    /// Post-render scanline (scanline 240)
    fn cycle_post_render(&mut self) {
        if self.cycles >= SCANLINE_END_CYCLE {
            self.scanline += 1;
            self.cycles = 0;
            self.pipeline_state = PipelineState::VerticalBlank;
        }
    }

    /// Vertical blanking scanlines (241-260)
    fn cycle_vblank(&mut self) {
        if self.cycles == 1 && self.scanline == VISIBLE_SCANLINES + 1 {
            self.is_vblank = true;
            self.frame_complete = true;
            if self.is_interrupting {
                self.nmi_pending = true;
            }
        }

        if self.cycles >= SCANLINE_END_CYCLE {
            self.scanline += 1;
            self.cycles = 0;
        }

        if self.scanline >= FRAME_END_SCANLINE {
            self.pipeline_state = PipelineState::PreRender;
            self.scanline = 0;
            self.is_even_frame = !self.is_even_frame;
        }
    }
}

// Rendering helpers
impl Ppu {
    /// Render a single pixel at the current cycle/scanline position
    #[inline]
    fn render_pixel(&mut self) {
        let x = self.cycles - 1;
        let y = self.scanline;

        let (bg_color, bg_opaque) = self.fetch_background_pixel(x);
        let (spr_color, spr_opaque, spr_foreground) = self.fetch_sprite_pixel(x, y, bg_opaque);

        // Priority multiplexer: determine final palette address
        let palette_addr = if (spr_foreground || !bg_opaque) && spr_opaque {
            spr_color
        } else if bg_opaque {
            bg_color
        } else {
            0 // universal background color
        };

        self.screen[y as usize * SCANLINE_VISIBLE_DOTS as usize + x as usize] =
            PALETTE[self.bus.read_palette(palette_addr) as usize];

        // Horizontal scroll increment at tile boundaries
        if self.is_rendering_enabled() {
            let x_fine = ((self.fine_x_scroll as i32 + x) % TILE_PIXELS) as u8;
            if x_fine == TILE_PIXELS as u8 - 1 {
                self.increment_horizontal_scroll();
            }
        }
    }

    /// Fetch background pixel color at position x
    /// Returns (palette_index, is_opaque)
    #[inline]
    fn fetch_background_pixel(&mut self, x: i32) -> (NesByte, bool) {
        if !self.is_showing_background {
            return (0, false);
        }

        let x_fine = ((self.fine_x_scroll as i32 + x) % TILE_PIXELS) as u8;

        if self.is_hiding_edge_background && x < TILE_PIXELS {
            return (0, false);
        }

        let mut bg_color = self.read_bg_pattern_bits(x_fine);
        let bg_opaque = bg_color != 0;
        bg_color |= self.read_bg_attribute_bits(x_fine);

        (bg_color, bg_opaque)
    }

    /// Read the 2-bit pattern (low bits) for the background tile at the current address.
    ///
    /// CHR bytes are identical for all 8 pixels in a tile row — fetch once when the
    /// cache is stale, then reuse for the remaining 7 pixels.
    #[inline]
    fn read_bg_pattern_bits(&mut self, x_fine: u8) -> NesByte {
        if x_fine == 0 || !self.bg_cache_valid {
            let tile_address = NAMETABLE_BASE | (self.data_address & NAMETABLE_ADDR_MASK);
            let tile = self.bus.read(tile_address);
            let fine_y = (self.data_address >> 12) & FINE_Y_MASK;
            let pattern_addr =
                (tile as NesAddr * TILE_BYTE_SIZE + fine_y) | (self.background_page << 12);
            self.bg_tile_lo = self.bus.read(pattern_addr);
            self.bg_tile_hi = self.bus.read(pattern_addr + CHR_PLANE_OFFSET);
        }
        let bit_pos = (TILE_PIXELS as u8 - 1) ^ x_fine;
        let lo = (self.bg_tile_lo >> bit_pos) & 1;
        let hi = (self.bg_tile_hi >> bit_pos) & 1;
        lo | (hi << 1)
    }

    /// Read the 2-bit attribute (high bits) for the background tile at the current address.
    ///
    /// Stable across the entire tile — fetch once and reuse. This is the second of the
    /// two background fetches per tile, so we set `bg_cache_valid = true` here.
    #[inline]
    fn read_bg_attribute_bits(&mut self, x_fine: u8) -> NesByte {
        if x_fine == 0 || !self.bg_cache_valid {
            let attr_addr = ATTR_TABLE_BASE
                | (self.data_address & LOOPY_NAMETABLE)
                | ((self.data_address >> 4) & ATTR_BLOCK_MASK)
                | ((self.data_address >> 2) & ATTR_COLUMN_MASK);
            let attribute = self.bus.read(attr_addr);
            let shift = ((self.data_address >> 4) & ATTR_QUADRANT_Y_BIT)
                | (self.data_address & ATTR_QUADRANT_X_BIT);
            self.bg_attr_bits = ((attribute >> shift as u8) & PALETTE_BITS_MASK) << 2;
            self.bg_cache_valid = true;
        }
        self.bg_attr_bits
    }

    /// Increment coarse X, wrapping into the next horizontal nametable
    /// ref: https://www.nesdev.org/wiki/PPU_scrolling#Coarse_X_increment
    #[inline]
    fn increment_horizontal_scroll(&mut self) {
        if (self.data_address & LOOPY_COARSE_X) == COARSE_X_MAX {
            self.data_address &= !LOOPY_COARSE_X;
            self.data_address ^= LOOPY_NAMETABLE_H;
        } else {
            self.data_address += 1;
        }
    }

    /// Fetch sprite pixel color at position (x, y)
    /// Returns (palette_index, is_opaque, is_foreground)
    #[inline]
    fn fetch_sprite_pixel(&mut self, x: i32, y: i32, bg_opaque: bool) -> (NesByte, bool, bool) {
        if !self.is_showing_sprites || (self.is_hiding_edge_sprites && x < TILE_PIXELS) {
            return (0, false, false);
        }

        let num_sprites = self.scanline_sprites.len();
        for idx in 0..num_sprites {
            let i = self.scanline_sprites[idx];
            let base = i as usize * OAM_ENTRY_SIZE;
            let spr_x = self.sprite_memory[base + OAM_X] as i32;

            if x < spr_x || x >= spr_x + TILE_PIXELS {
                continue;
            }

            let spr_y = self.sprite_memory[base + OAM_Y] as i32 + 1;
            let tile = self.sprite_memory[base + OAM_TILE];
            let attribute = self.sprite_memory[base + OAM_ATTR];

            let (x_shift, y_offset) = self.apply_sprite_flips(attribute, x - spr_x, y - spr_y);

            let address = self.sprite_pattern_address(tile, y_offset);

            let spr_color = self.read_pattern_color(address, x_shift);
            if spr_color == 0 {
                continue; // transparent pixel
            }

            let palette_index = self.apply_sprite_palette(spr_color, attribute);
            let foreground = attribute & SPRITE_PRIORITY == 0;

            // Sprite 0 hit detection (does not trigger at x=255)
            if !self.is_sprite_zero_hit
                && self.is_showing_background
                && i == 0
                && bg_opaque
                && x < 255
            {
                self.is_sprite_zero_hit = true;
            }

            return (palette_index, true, foreground);
        }

        (0, false, false)
    }

    /// Apply horizontal and vertical flip to sprite pixel coordinates
    #[inline]
    fn apply_sprite_flips(&self, attribute: NesByte, dx: i32, dy: i32) -> (u8, u16) {
        let sprite_height = if self.is_long_sprites {
            SPRITE_HEIGHT_TALL
        } else {
            SPRITE_HEIGHT_NORMAL
        };

        let mut x_shift = (dx % TILE_PIXELS) as u8;
        let mut y_offset = (dy % sprite_height) as u16;

        if attribute & SPRITE_FLIP_HORIZONTAL == 0 {
            x_shift ^= TILE_PIXELS as u8 - 1;
        }
        if attribute & SPRITE_FLIP_VERTICAL != 0 {
            y_offset ^= (sprite_height - 1) as u16;
        }

        (x_shift, y_offset)
    }

    /// Calculate the CHR ROM/RAM address for a sprite's pattern data
    #[inline]
    fn sprite_pattern_address(&self, tile: NesByte, y_offset: u16) -> NesAddr {
        if !self.is_long_sprites {
            // 8x8 sprites
            let mut addr = tile as NesAddr * TILE_BYTE_SIZE + y_offset;
            if self.sprite_page == CHR_PAGE_HIGH {
                addr += CHR_PAGE_OFFSET;
            }
            addr
        } else {
            // 8x16 sprites use a different layout:
            //   tile bit 0 → CHR page (0 or $1000)
            //   tile bits 1-7 → tile pair index (each pair = 32 bytes)
            //   y_offset 0-7 → top tile rows, 8-15 → bottom tile rows
            let row_in_tile = y_offset & 7;
            let bottom_tile_offset = (y_offset & TILE_PIXELS as u16) << 1; // 0 for top half, 16 for bottom half
            let tile_pair_base = (tile as NesAddr >> 1) * (TILE_BYTE_SIZE * 2);
            let chr_page = (tile as NesAddr & 1) << FINE_Y_SHIFT;
            tile_pair_base | bottom_tile_offset | row_in_tile | chr_page
        }
    }

    /// Read a 2-bit color from two CHR bitplanes at the given address and bit position.
    #[inline]
    fn read_pattern_color(&mut self, address: NesAddr, bit_pos: u8) -> NesByte {
        let lo = (self.bus.read(address) >> bit_pos) & 1;
        let hi = (self.bus.read(address + CHR_PLANE_OFFSET) >> bit_pos) & 1;
        lo | (hi << 1)
    }

    /// Combine sprite pattern color with palette bits from attribute
    #[inline]
    fn apply_sprite_palette(&self, pattern_color: NesByte, attribute: NesByte) -> NesByte {
        pattern_color
            | SPRITE_PALETTE_BASE // sprite palette starts at $10
            | ((attribute & SPRITE_PALETTE_MASK) << 2)
    }

    /// Increment the vertical scroll component of data_address
    /// ref: https://www.nesdev.org/wiki/PPU_scrolling#Wrapping_around
    fn increment_vertical_scroll(&mut self) {
        // fine_y / coarse_y are inputs to the cached pattern + attribute fetch addresses;
        // any mutation here invalidates the cache. (`copy_horizontal_scroll` at dot 257
        // also clears it, but that runs only when rendering is enabled — explicit
        // invalidation here keeps us safe if rendering toggles between dot 256 and 257.)
        self.bg_cache_valid = false;
        if (self.data_address & LOOPY_FINE_Y) != LOOPY_FINE_Y {
            self.data_address += LOOPY_FINE_Y_INC;
            return;
        }

        // Fine Y overflowed (was 7) — reset to 0 and increment coarse Y
        self.data_address &= !LOOPY_FINE_Y;
        let mut coarse_y = (self.data_address & LOOPY_COARSE_Y) >> 5;
        if coarse_y == COARSE_Y_LAST_ROW {
            coarse_y = 0;
            self.data_address ^= LOOPY_NAMETABLE_V; // switch vertical nametable
        } else if coarse_y == COARSE_Y_MAX {
            coarse_y = 0; // in attribute table area, wrap without nametable switch
        } else {
            coarse_y += 1;
        }
        self.data_address = (self.data_address & !LOOPY_COARSE_Y) | (coarse_y << 5);
    }

    /// Find sprites on the next scanline
    fn evaluate_sprites(&mut self) {
        self.scanline_sprites.clear();

        let sprite_height = if self.is_long_sprites {
            SPRITE_HEIGHT_TALL
        } else {
            SPRITE_HEIGHT_NORMAL
        };
        for (i, entry) in self.sprite_memory.chunks(OAM_ENTRY_SIZE).enumerate() {
            let diff = self.scanline - entry[OAM_Y] as i32;
            if diff < 0 || diff >= sprite_height {
                continue;
            }
            if self.scanline_sprites.len() >= MAX_SPRITES_PER_SCANLINE {
                self.is_sprite_overflow = true;
                break;
            }
            self.scanline_sprites.push(i as NesByte);
        }
    }
}
