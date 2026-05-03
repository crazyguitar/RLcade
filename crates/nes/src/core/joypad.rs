use super::common::NesByte;

// NES standard controller button bit masks
// ref: https://www.nesdev.org/wiki/Standard_controller
// Buttons are read in this order: A, B, Select, Start, Up, Down, Left, Right
pub const BUTTON_A: NesByte = 1 << 0;
pub const BUTTON_B: NesByte = 1 << 1;
pub const BUTTON_SELECT: NesByte = 1 << 2;
pub const BUTTON_START: NesByte = 1 << 3;
pub const BUTTON_UP: NesByte = 1 << 4;
pub const BUTTON_DOWN: NesByte = 1 << 5;
pub const BUTTON_LEFT: NesByte = 1 << 6;
pub const BUTTON_RIGHT: NesByte = 1 << 7;

const NUM_BUTTONS: u8 = 8;
pub const NUM_PLAYERS: usize = 2;
pub const PLAYER1: usize = 0;
pub const PLAYER2: usize = 1;

/// NES standard controller
/// ref: https://www.nesdev.org/wiki/Standard_controller
///
/// The controller uses a parallel-in, serial-out shift register.
/// Writing 1 then 0 to $4016 latches the current button state.
/// Each subsequent read from $4016/$4017 returns one button bit.
pub struct Joypad {
    /// Current button state (updated by the emulator from input)
    buttons: NesByte,
    /// Shift register position (0-7 = button bits, 8+ = always 1)
    shift: u8,
    /// Strobe mode: while true, reads always return button A
    strobe: bool,
}

impl Default for Joypad {
    fn default() -> Self {
        Self::new()
    }
}
impl Joypad {
    pub fn new() -> Self {
        Self {
            buttons: 0,
            shift: 0,
            strobe: false,
        }
    }

    /// Set the current button state (called by the emulator from input handling)
    pub fn set_buttons(&mut self, state: NesByte) {
        self.buttons = state;
    }

    /// Get the current button state
    pub fn buttons(&self) -> NesByte {
        self.buttons
    }

    /// Write to the strobe register ($4016 bit 0)
    /// Writing 1 enables strobe mode (continuously reloads shift register).
    /// Writing 0 disables strobe and latches the current button state.
    pub fn write_strobe(&mut self, value: NesByte) {
        let new_strobe = value & 1 != 0;
        // When strobe transitions from 1 to 0, latch button state
        if self.strobe && !new_strobe {
            self.shift = 0;
        }
        self.strobe = new_strobe;
    }

    /// Serial read: return one button bit per read
    /// While strobe is active, always returns the state of button A.
    /// After 8 reads, returns 1 (open bus behavior on standard controller).
    pub fn read(&mut self) -> NesByte {
        if self.strobe {
            // While strobe is high, always return button A
            return self.buttons & 1;
        }

        if self.shift < NUM_BUTTONS {
            let bit = (self.buttons >> self.shift) & 1;
            self.shift += 1;
            bit
        } else {
            1 // after 8 reads, returns 1
        }
    }
}
