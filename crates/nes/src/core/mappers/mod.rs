pub mod cnrom;
pub mod mapper;
pub mod nrom;
pub mod sxrom;
pub mod txrom;
pub mod uxrom;

pub use cnrom::Cnrom;
pub use nrom::Nrom;
pub use sxrom::Sxrom;
pub use txrom::Txrom;
pub use uxrom::Uxrom;

use crate::core::cartridge::Cartridge;
use crate::core::common::{NesAddr, NesByte};
use mapper::Mapper;

/// Concrete mapper dispatched via `match` on the small set of supported
/// variants. Replaces `Box<dyn Mapper>` to eliminate vtable/indirect-call
/// cost from the hottest path (PRG + CHR reads).
pub enum MapperKind {
    Nrom(Nrom),
    Sxrom(Sxrom),
    Uxrom(Uxrom),
    Cnrom(Cnrom),
    Txrom(Txrom),
}

macro_rules! dispatch {
    ($self:expr, $m:ident, $e:expr) => {
        match $self {
            MapperKind::Nrom($m) => $e,
            MapperKind::Sxrom($m) => $e,
            MapperKind::Uxrom($m) => $e,
            MapperKind::Cnrom($m) => $e,
            MapperKind::Txrom($m) => $e,
        }
    };
}

impl MapperKind {
    #[inline]
    pub fn read_prg(&self, addr: NesAddr) -> NesByte {
        dispatch!(self, m, m.read_prg(addr))
    }

    #[inline]
    pub fn write_prg(&mut self, addr: NesAddr, value: NesByte) {
        dispatch!(self, m, m.write_prg(addr, value))
    }

    #[inline]
    pub fn read_chr(&self, addr: NesAddr) -> NesByte {
        dispatch!(self, m, m.read_chr(addr))
    }

    #[inline]
    pub fn write_chr(&mut self, addr: NesAddr, value: NesByte) {
        dispatch!(self, m, m.write_chr(addr, value))
    }

    #[inline]
    pub fn cartridge(&self) -> &Cartridge {
        dispatch!(self, m, m.cartridge())
    }

    #[inline]
    pub fn has_extended_ram(&self) -> bool {
        dispatch!(self, m, m.has_extended_ram())
    }

    #[inline]
    pub fn scanline_counter(&mut self) {
        dispatch!(self, m, m.scanline_counter())
    }

    #[inline]
    pub fn take_irq(&mut self) -> bool {
        dispatch!(self, m, m.take_irq())
    }

    #[inline]
    pub fn take_mirroring(&mut self) -> Option<u8> {
        dispatch!(self, m, m.take_mirroring())
    }

    #[inline]
    pub fn reset(&mut self) {
        dispatch!(self, m, m.reset())
    }
}
