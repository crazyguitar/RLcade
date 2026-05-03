use super::common::NesByte;

// NES APU (Audio Processing Unit)
// ref: https://www.nesdev.org/wiki/APU

// CPU clock rate (NTSC)
const CPU_FREQUENCY: f32 = 1_789_773.0;

// Audio output sample rate
pub const SAMPLE_RATE: u32 = 44_100;

// Length counter lookup table (32 entries, indexed by bits 7-3 of $4003/$4007/$400B/$400F)
// ref: https://www.nesdev.org/wiki/APU_Length_Counter
const LENGTH_TABLE: [u8; 32] = [
    10, 254, 20, 2, 40, 4, 80, 6, 160, 8, 60, 10, 14, 12, 26, 14, 12, 16, 24, 18, 48, 20, 96, 22,
    192, 24, 72, 26, 16, 28, 32, 30,
];

// Duty cycle waveforms for pulse channels (8-step sequences)
// ref: https://www.nesdev.org/wiki/APU_Pulse
const DUTY_TABLE: [[u8; 8]; 4] = [
    [0, 1, 0, 0, 0, 0, 0, 0], // 12.5%
    [0, 1, 1, 0, 0, 0, 0, 0], // 25%
    [0, 1, 1, 1, 1, 0, 0, 0], // 50%
    [1, 0, 0, 1, 1, 1, 1, 1], // 75% (inverted 25%)
];

// Triangle channel waveform (32-step sequence)
// ref: https://www.nesdev.org/wiki/APU_Triangle
const TRIANGLE_TABLE: [u8; 32] = [
    15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12,
    13, 14, 15,
];

// Noise channel period lookup table (NTSC)
// ref: https://www.nesdev.org/wiki/APU_Noise
const NOISE_TABLE: [u16; 16] = [
    4, 8, 16, 32, 64, 96, 128, 160, 202, 254, 380, 508, 762, 1016, 2034, 4068,
];

// DMC rate lookup table (NTSC, in CPU cycles)
// ref: https://www.nesdev.org/wiki/APU_DMC
const DMC_TABLE: [u16; 16] = [
    428, 380, 340, 320, 286, 254, 226, 214, 190, 160, 142, 128, 106, 84, 72, 54,
];

// Frame counter step boundaries (in CPU cycles).
// The wiki lists APU cycle counts (3728.5, 7456.5, …) which are half of CPU cycles.
// Since frame_cycle increments once per CPU cycle, we use the doubled (CPU) values.
// ref: https://www.nesdev.org/wiki/APU_Frame_Counter
const FRAME_STEP_1: u16 = 7457; // quarter frame (envelope + linear counter)
const FRAME_STEP_2: u16 = 14913; // half frame (sweep + length) + quarter frame
const FRAME_STEP_3: u16 = 22371; // quarter frame
const FRAME_STEP_4: u16 = 29829; // 4-step: half + quarter + IRQ + reset; 5-step: nothing
const FRAME_STEP_5: u16 = 37281; // 5-step only: half + quarter + reset

// Mixer lookup tables
// ref: https://www.nesdev.org/wiki/APU_Mixer
const PULSE_TABLE_SIZE: usize = 31;
const TND_TABLE_SIZE: usize = 203;

// Mixer formula constants from APU Mixer reference
// pulse_out = PULSE_NUMERATOR / (PULSE_DENOMINATOR / pulse_index + MIXER_OFFSET)
// tnd_out   = TND_NUMERATOR   / (TND_DENOMINATOR   / tnd_index   + MIXER_OFFSET)
const PULSE_NUMERATOR: f32 = 95.52;
const PULSE_DENOMINATOR: f32 = 8128.0;
const TND_NUMERATOR: f32 = 163.67;
const TND_DENOMINATOR: f32 = 24329.0;
const MIXER_OFFSET: f32 = 100.0;

// Envelope constants
const ENVELOPE_MAX: u8 = 15; // max decay level (4-bit)

// Timer period constants (11-bit timer)
const TIMER_HIGH_MASK: u16 = 0x700; // bits 8-10 of timer period
const TIMER_LOW_MASK: u16 = 0x0FF; // bits 0-7 of timer period
const TIMER_HIGH_BITS: u16 = 0x07; // valid bits from high byte write
const TIMER_HIGH_SHIFT: u8 = 8;

/// Update an 11-bit timer period: keep the low 8 bits, replace the high 3 bits from value
fn set_timer_high(timer_period: u16, value: u8) -> u16 {
    (timer_period & TIMER_LOW_MASK) | ((value as u16 & TIMER_HIGH_BITS) << TIMER_HIGH_SHIFT)
}

/// Update an 11-bit timer period: keep the high 3 bits, replace the low 8 bits from value
fn set_timer_low(timer_period: u16, value: u8) -> u16 {
    (timer_period & TIMER_HIGH_MASK) | value as u16
}

// Pulse sweep constants
const SWEEP_PERIOD_MIN: u16 = 8; // periods below this are muted
const SWEEP_PERIOD_MAX: u16 = 0x7FF; // 11-bit max timer period

// Triangle sequencer
const TRIANGLE_SEQ_MASK: u8 = 31; // 32-step sequence (0-31)
const TRIANGLE_ULTRASONIC_MIN: u16 = 2; // silence below this period

// Noise LFSR
const NOISE_LFSR_INIT: u16 = 1; // initial shift register value
const NOISE_LFSR_FEEDBACK_BIT: u16 = 14; // feedback inserted at bit 14
const NOISE_SHORT_MODE_BIT: u16 = 6; // XOR with bit 6 in short mode
const NOISE_LONG_MODE_BIT: u16 = 1; // XOR with bit 1 in long mode

// DMC constants
const DMC_SAMPLE_BASE: u16 = 0xC000; // sample address = $C000 + A*64
const DMC_ADDR_STEP: u16 = 64; // address granularity
const DMC_LENGTH_STEP: u16 = 16; // length granularity
const DMC_OUTPUT_MASK: u8 = 0x7F; // 7-bit output level
const DMC_OUTPUT_MAX: u8 = 125; // max level before +2 increment
const DMC_OUTPUT_MIN: u8 = 2; // min level before -2 decrement
const DMC_OUTPUT_DELTA: u8 = 2; // increment/decrement step
const DMC_ADDR_WRAP: u16 = 0x8000; // address wraps to $8000-$FFFF
const DMC_BITS_PER_SAMPLE: u8 = 8; // bits per sample byte

// Status register ($4015) channel bits
const STATUS_PULSE1: u8 = 0x01;
const STATUS_PULSE2: u8 = 0x02;
const STATUS_TRIANGLE: u8 = 0x04;
const STATUS_NOISE: u8 = 0x08;
const STATUS_DMC: u8 = 0x10;
const STATUS_FRAME_IRQ: u8 = 0x40;
const STATUS_DMC_IRQ: u8 = 0x80;

// Duty cycle sequencer
const DUTY_STEPS: u8 = 8; // 8-step duty cycle sequence
const DUTY_MASK: u8 = 0x03; // 2-bit duty select (bits 6-7)
const DUTY_SHIFT: u8 = 6;

// Pulse/Noise control register bits (0x4000/0x4004/0x400C)
const CHANNEL_LENGTH_HALT: u8 = 0x20;
const CHANNEL_CONSTANT_VOLUME: u8 = 0x10;
const CHANNEL_VOLUME_MASK: u8 = 0x0F;

// Pulse sweep register bits (0x4001/0x4005)
const SWEEP_ENABLED: u8 = 0x80;
const SWEEP_NEGATE: u8 = 0x08;
const SWEEP_PERIOD_MASK: u8 = 0x07;
const SWEEP_SHIFT_MASK: u8 = 0x07;
const SWEEP_PERIOD_SHIFT: u8 = 4;

// Triangle linear counter (0x4008)
const TRIANGLE_LENGTH_HALT: u8 = 0x80;
const TRIANGLE_LINEAR_LOAD_MASK: u8 = 0x7F;

// Noise mode register (0x400E)
const NOISE_MODE_FLAG: u8 = 0x80;
const NOISE_PERIOD_INDEX_MASK: u8 = 0x0F;

// DMC control register (0x4010)
const DMC_IRQ_ENABLE: u8 = 0x80;
const DMC_LOOP_FLAG: u8 = 0x40;
const DMC_RATE_INDEX_MASK: u8 = 0x0F;

// Frame counter register (0x4017)
const FRAME_MODE_5STEP: u8 = 0x80;
const FRAME_IRQ_INHIBIT: u8 = 0x40;

// SDL2 audio buffer size (smaller = lower latency, fewer underrun gaps)
pub const AUDIO_BUFFER_SAMPLES: u16 = 512;

/// Reusable length-counter component shared by Pulse, Triangle, and Noise channels.
struct LengthCounter {
    enabled: bool,
    counter: u8,
    halt: bool,
}

impl LengthCounter {
    fn new() -> Self {
        Self {
            enabled: false,
            counter: 0,
            halt: false,
        }
    }

    fn clock(&mut self) {
        if !self.halt && self.counter > 0 {
            self.counter -= 1;
        }
    }

    fn set_enabled(&mut self, enabled: bool) {
        self.enabled = enabled;
        if !enabled {
            self.counter = 0;
        }
    }
}

/// Envelope unit shared by pulse and noise channels
/// ref: https://www.nesdev.org/wiki/APU_Envelope
struct Envelope {
    start: bool,
    loop_flag: bool,
    constant_volume: bool,
    volume: u8,
    divider: u8,
    decay: u8,
}

impl Envelope {
    fn new() -> Self {
        Self {
            start: false,
            loop_flag: false,
            constant_volume: false,
            volume: 0,
            divider: 0,
            decay: 0,
        }
    }

    fn clock(&mut self) {
        if self.start {
            self.start = false;
            self.decay = ENVELOPE_MAX;
            self.divider = self.volume;
        } else if self.divider == 0 {
            self.divider = self.volume;
            self.clock_decay();
        } else {
            self.divider -= 1;
        }
    }

    fn clock_decay(&mut self) {
        if self.decay > 0 {
            self.decay -= 1;
        } else if self.loop_flag {
            self.decay = ENVELOPE_MAX;
        }
    }

    #[inline]
    fn output(&self) -> u8 {
        if self.constant_volume {
            self.volume
        } else {
            self.decay
        }
    }
}

/// Sweep unit for pulse channels
/// ref: https://www.nesdev.org/wiki/APU_Sweep
struct Sweep {
    enabled: bool,
    negate: bool,
    reload: bool,
    period: u8,
    shift: u8,
    divider: u8,
    /// Pulse 1 uses ones' complement negate, Pulse 2 uses two's complement
    is_pulse1: bool,
}

impl Sweep {
    fn new(is_pulse1: bool) -> Self {
        Self {
            enabled: false,
            negate: false,
            reload: false,
            period: 0,
            shift: 0,
            divider: 0,
            is_pulse1,
        }
    }

    fn target_period(&self, current_period: u16) -> u16 {
        let delta = current_period >> self.shift;
        if self.negate {
            if self.is_pulse1 {
                current_period.wrapping_sub(delta + 1) // ones' complement
            } else {
                current_period.wrapping_sub(delta) // two's complement
            }
        } else {
            current_period + delta
        }
    }

    fn is_muting(&self, current_period: u16) -> bool {
        current_period < SWEEP_PERIOD_MIN || self.target_period(current_period) > SWEEP_PERIOD_MAX
    }

    fn clock(&mut self, current_period: &mut u16) {
        let divider_expired = self.divider == 0;
        let sweep_active = self.enabled && self.shift > 0;
        let valid_period = !self.is_muting(*current_period);

        if divider_expired && sweep_active && valid_period {
            *current_period = self.target_period(*current_period);
        }
        if self.divider == 0 || self.reload {
            self.divider = self.period;
            self.reload = false;
        } else {
            self.divider -= 1;
        }
    }
}

/// Pulse channel (two instances: Pulse 1 and Pulse 2)
/// ref: https://www.nesdev.org/wiki/APU_Pulse
struct Pulse {
    length: LengthCounter,
    duty: u8,
    duty_pos: u8,
    timer: u16,
    timer_period: u16,
    envelope: Envelope,
    sweep: Sweep,
}

impl Pulse {
    fn new(is_pulse1: bool) -> Self {
        Self {
            length: LengthCounter::new(),
            duty: 0,
            duty_pos: 0,
            timer: 0,
            timer_period: 0,
            envelope: Envelope::new(),
            sweep: Sweep::new(is_pulse1),
        }
    }

    fn write_control(&mut self, value: NesByte) {
        self.duty = (value >> DUTY_SHIFT) & DUTY_MASK;
        self.length.halt = value & CHANNEL_LENGTH_HALT != 0;
        self.envelope.loop_flag = self.length.halt;
        self.envelope.constant_volume = value & CHANNEL_CONSTANT_VOLUME != 0;
        self.envelope.volume = value & CHANNEL_VOLUME_MASK;
    }

    fn write_sweep(&mut self, value: NesByte) {
        self.sweep.enabled = value & SWEEP_ENABLED != 0;
        self.sweep.period = (value >> SWEEP_PERIOD_SHIFT) & SWEEP_PERIOD_MASK;
        self.sweep.negate = value & SWEEP_NEGATE != 0;
        self.sweep.shift = value & SWEEP_SHIFT_MASK;
        self.sweep.reload = true;
    }

    fn write_timer_low(&mut self, value: NesByte) {
        self.timer_period = set_timer_low(self.timer_period, value);
    }

    fn write_timer_high(&mut self, value: NesByte) {
        self.timer_period = set_timer_high(self.timer_period, value);
        if self.length.enabled {
            self.length.counter = LENGTH_TABLE[(value >> 3) as usize];
        }
        self.duty_pos = 0;
        self.envelope.start = true;
    }

    #[inline]
    fn clock_timer(&mut self) {
        if self.timer == 0 {
            self.timer = self.timer_period;
            self.duty_pos = (self.duty_pos + 1) & (DUTY_STEPS - 1);
        } else {
            self.timer -= 1;
        }
    }

    #[inline]
    fn output(&self) -> u8 {
        if !self.length.enabled
            || self.length.counter == 0
            || DUTY_TABLE[self.duty as usize][self.duty_pos as usize] == 0
            || self.sweep.is_muting(self.timer_period)
        {
            0
        } else {
            self.envelope.output()
        }
    }
}

/// Triangle channel
/// ref: https://www.nesdev.org/wiki/APU_Triangle
struct Triangle {
    length: LengthCounter,
    sequencer_pos: u8,
    timer: u16,
    timer_period: u16,
    linear_counter: u8,
    linear_reload: u8,
    linear_reload_flag: bool,
}

impl Triangle {
    fn new() -> Self {
        Self {
            length: LengthCounter::new(),
            sequencer_pos: 0,
            timer: 0,
            timer_period: 0,
            linear_counter: 0,
            linear_reload: 0,
            linear_reload_flag: false,
        }
    }

    fn write_linear(&mut self, value: NesByte) {
        self.length.halt = value & TRIANGLE_LENGTH_HALT != 0;
        self.linear_reload = value & TRIANGLE_LINEAR_LOAD_MASK;
    }

    fn write_timer_low(&mut self, value: NesByte) {
        self.timer_period = set_timer_low(self.timer_period, value);
    }

    fn write_timer_high(&mut self, value: NesByte) {
        self.timer_period = set_timer_high(self.timer_period, value);
        if self.length.enabled {
            self.length.counter = LENGTH_TABLE[(value >> 3) as usize];
        }
        self.linear_reload_flag = true;
    }

    #[inline]
    fn clock_timer(&mut self) {
        if self.timer == 0 {
            self.timer = self.timer_period;
            if self.length.counter > 0 && self.linear_counter > 0 {
                self.sequencer_pos = (self.sequencer_pos + 1) & TRIANGLE_SEQ_MASK;
            }
        } else {
            self.timer -= 1;
        }
    }

    fn clock_linear(&mut self) {
        if self.linear_reload_flag {
            self.linear_counter = self.linear_reload;
        } else if self.linear_counter > 0 {
            self.linear_counter -= 1;
        }
        if !self.length.halt {
            self.linear_reload_flag = false;
        }
    }

    /// Triangle channel always outputs current sequencer position.
    /// When length/linear counter expires, the sequencer freezes but the DAC
    /// holds its last value (avoids audio pops).
    /// ref: https://www.nesdev.org/wiki/APU_Triangle
    #[inline]
    fn output(&self) -> u8 {
        TRIANGLE_TABLE[self.sequencer_pos as usize]
    }
}

/// Noise channel
/// ref: https://www.nesdev.org/wiki/APU_Noise
struct Noise {
    length: LengthCounter,
    mode: bool,
    timer: u16,
    timer_period: u16,
    shift_register: u16,
    envelope: Envelope,
}

impl Noise {
    fn new() -> Self {
        Self {
            length: LengthCounter::new(),
            mode: false,
            timer: 0,
            timer_period: 0,
            shift_register: NOISE_LFSR_INIT,
            envelope: Envelope::new(),
        }
    }

    fn write_control(&mut self, value: NesByte) {
        self.length.halt = value & CHANNEL_LENGTH_HALT != 0;
        self.envelope.loop_flag = self.length.halt;
        self.envelope.constant_volume = value & CHANNEL_CONSTANT_VOLUME != 0;
        self.envelope.volume = value & CHANNEL_VOLUME_MASK;
    }

    fn write_period(&mut self, value: NesByte) {
        self.mode = value & NOISE_MODE_FLAG != 0;
        self.timer_period = NOISE_TABLE[(value & NOISE_PERIOD_INDEX_MASK) as usize];
    }

    fn write_length(&mut self, value: NesByte) {
        if self.length.enabled {
            self.length.counter = LENGTH_TABLE[(value >> 3) as usize];
        }
        self.envelope.start = true;
    }

    #[inline]
    fn clock_timer(&mut self) {
        if self.timer == 0 {
            self.timer = self.timer_period;
            // LFSR feedback: XOR bit 0 with bit 6 (mode=1) or bit 1 (mode=0)
            let feedback_bit = if self.mode {
                NOISE_SHORT_MODE_BIT
            } else {
                NOISE_LONG_MODE_BIT
            };
            let feedback = (self.shift_register & 1) ^ ((self.shift_register >> feedback_bit) & 1);
            self.shift_register >>= 1;
            self.shift_register |= feedback << NOISE_LFSR_FEEDBACK_BIT;
        } else {
            self.timer -= 1;
        }
    }

    #[inline]
    fn output(&self) -> u8 {
        if !self.length.enabled || self.length.counter == 0 || self.shift_register & 1 != 0 {
            0
        } else {
            self.envelope.output()
        }
    }
}

/// DMC (Delta Modulation Channel)
/// ref: https://www.nesdev.org/wiki/APU_DMC
struct Dmc {
    enabled: bool,
    irq_enabled: bool,
    irq_pending: bool,
    loop_flag: bool,
    timer: u16,
    timer_period: u16,
    output_level: u8,
    sample_address: u16,
    sample_length: u16,
    current_address: u16,
    bytes_remaining: u16,
    sample_buffer: Option<u8>,
    shift_register: u8,
    bits_remaining: u8,
    silence: bool,
}

impl Dmc {
    fn new() -> Self {
        Self {
            enabled: false,
            irq_enabled: false,
            irq_pending: false,
            loop_flag: false,
            timer: 0,
            timer_period: 0,
            output_level: 0,
            sample_address: DMC_SAMPLE_BASE,
            sample_length: 1,
            current_address: DMC_SAMPLE_BASE,
            bytes_remaining: 0,
            sample_buffer: None,
            shift_register: 0,
            bits_remaining: 0,
            silence: true,
        }
    }

    fn write_control(&mut self, value: NesByte) {
        self.irq_enabled = value & DMC_IRQ_ENABLE != 0;
        self.loop_flag = value & DMC_LOOP_FLAG != 0;
        self.timer_period = DMC_TABLE[(value & DMC_RATE_INDEX_MASK) as usize];
        if !self.irq_enabled {
            self.irq_pending = false;
        }
    }

    fn write_output(&mut self, value: NesByte) {
        self.output_level = value & DMC_OUTPUT_MASK;
    }

    fn write_address(&mut self, value: NesByte) {
        self.sample_address = DMC_SAMPLE_BASE | (value as u16 * DMC_ADDR_STEP);
    }

    fn write_length(&mut self, value: NesByte) {
        self.sample_length = (value as u16 * DMC_LENGTH_STEP) + 1;
    }

    fn restart(&mut self) {
        self.current_address = self.sample_address;
        self.bytes_remaining = self.sample_length;
    }

    #[inline]
    fn clock_timer(&mut self) {
        if self.timer == 0 {
            self.timer = self.timer_period;
            self.clock_output();
        } else {
            self.timer -= 1;
        }
    }

    fn clock_output(&mut self) {
        if !self.silence {
            self.update_output_level();
            self.shift_register >>= 1;
        }
        self.bits_remaining = self.bits_remaining.saturating_sub(1);
        self.reload_shift_register();
    }

    /// Adjust output level based on current shift register bit
    fn update_output_level(&mut self) {
        if self.shift_register & 1 != 0 {
            if self.output_level <= DMC_OUTPUT_MAX {
                self.output_level += DMC_OUTPUT_DELTA;
            }
        } else if self.output_level >= DMC_OUTPUT_MIN {
            self.output_level -= DMC_OUTPUT_DELTA;
        }
    }

    /// When all bits consumed, load next sample byte into shift register
    fn reload_shift_register(&mut self) {
        if self.bits_remaining > 0 {
            return;
        }
        self.bits_remaining = DMC_BITS_PER_SAMPLE;
        match self.sample_buffer.take() {
            Some(buffer) => {
                self.silence = false;
                self.shift_register = buffer;
            }
            None => self.silence = true,
        }
    }

    /// Accept a sample byte from CPU memory and advance address
    fn load_sample(&mut self, value: NesByte) {
        self.sample_buffer = Some(value);
        self.advance_address();
        self.bytes_remaining -= 1;
        if self.bytes_remaining == 0 {
            self.handle_sample_end();
        }
    }

    /// Advance DMC read address; on overflow past $FFFF, wrap to $8000.
    /// ref: https://www.nesdev.org/wiki/APU_DMC
    fn advance_address(&mut self) {
        let next = self.current_address.wrapping_add(1);
        self.current_address = if next == 0 { DMC_ADDR_WRAP } else { next };
    }

    /// Handle end of sample: loop or trigger IRQ
    fn handle_sample_end(&mut self) {
        if self.loop_flag {
            self.restart();
        } else if self.irq_enabled {
            self.irq_pending = true;
        }
    }

    #[inline]
    fn output(&self) -> u8 {
        self.output_level
    }
}

/// NES APU
pub struct Apu {
    pulse1: Pulse,
    pulse2: Pulse,
    triangle: Triangle,
    noise: Noise,
    dmc: Dmc,

    /// Frame counter cycle count (CPU cycles)
    frame_cycle: u16,
    /// Frame counter mode: false = 4-step, true = 5-step
    frame_mode_5step: bool,
    /// Frame counter IRQ inhibit flag
    frame_irq_inhibit: bool,
    /// Frame counter IRQ pending
    pub frame_irq_pending: bool,
    /// Toggles every CPU cycle; pulse/noise/DMC timers clock when true (APU half-rate)
    apu_cycle_odd: bool,

    /// Audio sample accumulation
    sample_timer: f32,
    sample_period: f32,
    /// Audio output buffer
    buffer: Vec<f32>,

    /// Mixer lookup tables
    pulse_table: [f32; PULSE_TABLE_SIZE],
    tnd_table: [f32; TND_TABLE_SIZE],
}

impl Default for Apu {
    fn default() -> Self {
        Self::new()
    }
}
impl Apu {
    pub fn new() -> Self {
        // Build mixer lookup tables
        // ref: https://www.nesdev.org/wiki/APU_Mixer
        let mut pulse_table = [0.0; PULSE_TABLE_SIZE];
        for (i, entry) in pulse_table.iter_mut().enumerate().skip(1) {
            *entry = PULSE_NUMERATOR / (PULSE_DENOMINATOR / i as f32 + MIXER_OFFSET);
        }

        let mut tnd_table = [0.0; TND_TABLE_SIZE];
        for (i, entry) in tnd_table.iter_mut().enumerate().skip(1) {
            *entry = TND_NUMERATOR / (TND_DENOMINATOR / i as f32 + MIXER_OFFSET);
        }

        Self {
            pulse1: Pulse::new(true),
            pulse2: Pulse::new(false),
            triangle: Triangle::new(),
            noise: Noise::new(),
            dmc: Dmc::new(),
            frame_cycle: 0,
            frame_mode_5step: false,
            frame_irq_inhibit: false,
            frame_irq_pending: false,
            apu_cycle_odd: false,
            sample_timer: 0.0,
            sample_period: CPU_FREQUENCY / SAMPLE_RATE as f32,
            buffer: Vec::with_capacity(SAMPLE_RATE as usize / 30),
            pulse_table,
            tnd_table,
        }
    }

    /// Write to APU register ($4000-$4017)
    pub fn write(&mut self, addr: u16, value: NesByte) {
        match addr {
            // Pulse 1
            0x4000 => self.pulse1.write_control(value),
            0x4001 => self.pulse1.write_sweep(value),
            0x4002 => self.pulse1.write_timer_low(value),
            0x4003 => self.pulse1.write_timer_high(value),
            // Pulse 2
            0x4004 => self.pulse2.write_control(value),
            0x4005 => self.pulse2.write_sweep(value),
            0x4006 => self.pulse2.write_timer_low(value),
            0x4007 => self.pulse2.write_timer_high(value),
            // Triangle
            0x4008 => self.triangle.write_linear(value),
            0x400A => self.triangle.write_timer_low(value),
            0x400B => self.triangle.write_timer_high(value),
            // Noise
            0x400C => self.noise.write_control(value),
            0x400E => self.noise.write_period(value),
            0x400F => self.noise.write_length(value),
            // DMC
            0x4010 => self.dmc.write_control(value),
            0x4011 => self.dmc.write_output(value),
            0x4012 => self.dmc.write_address(value),
            0x4013 => self.dmc.write_length(value),
            // Status
            0x4015 => self.write_status(value),
            // Frame counter
            0x4017 => self.write_frame_counter(value),
            _ => {}
        }
    }

    /// Read APU status ($4015)
    #[rustfmt::skip]
    pub fn read_status(&mut self) -> NesByte {
        let mut status = 0u8;
        if self.pulse1.length.counter > 0 { status |= STATUS_PULSE1; }
        if self.pulse2.length.counter > 0 { status |= STATUS_PULSE2; }
        if self.triangle.length.counter > 0 { status |= STATUS_TRIANGLE; }
        if self.noise.length.counter > 0 { status |= STATUS_NOISE; }
        if self.dmc.bytes_remaining > 0 { status |= STATUS_DMC; }
        if self.frame_irq_pending { status |= STATUS_FRAME_IRQ; }
        if self.dmc.irq_pending { status |= STATUS_DMC_IRQ; }
        // Reading $4015 clears frame IRQ flag
        self.frame_irq_pending = false;
        status
    }

    /// Write to status register ($4015) — enable/disable channels
    fn write_status(&mut self, value: NesByte) {
        self.pulse1.length.set_enabled(value & STATUS_PULSE1 != 0);
        self.pulse2.length.set_enabled(value & STATUS_PULSE2 != 0);
        self.triangle
            .length
            .set_enabled(value & STATUS_TRIANGLE != 0);
        self.noise.length.set_enabled(value & STATUS_NOISE != 0);
        self.set_dmc_enabled(value & STATUS_DMC != 0);
        self.dmc.irq_pending = false;
    }

    /// Enable/disable DMC; clear remaining bytes when disabled, restart when enabled with no bytes
    fn set_dmc_enabled(&mut self, enabled: bool) {
        self.dmc.enabled = enabled;
        if !enabled {
            self.dmc.bytes_remaining = 0;
        } else if self.dmc.bytes_remaining == 0 {
            self.dmc.restart();
        }
    }

    /// Write to frame counter register ($4017)
    fn write_frame_counter(&mut self, value: NesByte) {
        self.frame_mode_5step = value & FRAME_MODE_5STEP != 0;
        self.frame_irq_inhibit = value & FRAME_IRQ_INHIBIT != 0;
        if self.frame_irq_inhibit {
            self.frame_irq_pending = false;
        }
        self.frame_cycle = 0;
        // 5-step mode immediately clocks all units
        if self.frame_mode_5step {
            self.clock_quarter_frame();
            self.clock_half_frame();
        }
    }

    /// Advance APU by one CPU cycle
    #[inline]
    pub fn clock(&mut self) {
        // Triangle timer clocks at CPU rate
        self.triangle.clock_timer();

        // Pulse, noise, and DMC timers clock at APU rate (every other CPU cycle)
        self.apu_cycle_odd = !self.apu_cycle_odd;
        if self.apu_cycle_odd {
            self.pulse1.clock_timer();
            self.pulse2.clock_timer();
            self.noise.clock_timer();
            self.dmc.clock_timer();
        }

        // Frame counter
        self.clock_frame_counter();

        // Generate audio sample
        self.sample_timer += 1.0;
        if self.sample_timer >= self.sample_period {
            self.sample_timer -= self.sample_period;
            let sample = self.mix();
            self.buffer.push(sample);
        }

        self.frame_cycle += 1;
    }

    /// Clock the frame counter and trigger quarter/half frame events
    fn clock_frame_counter(&mut self) {
        if self.frame_mode_5step {
            self.clock_frame_counter_5step();
        } else {
            self.clock_frame_counter_4step();
        }
    }

    /// 5-step mode: quarter/half frame events, no IRQ
    fn clock_frame_counter_5step(&mut self) {
        match self.frame_cycle {
            FRAME_STEP_1 | FRAME_STEP_3 => self.clock_quarter_frame(),
            FRAME_STEP_2 => self.clock_half_and_quarter_frame(),
            FRAME_STEP_4 => {} // 5-step mode skips step 4
            FRAME_STEP_5 => self.reset_frame_cycle(),
            _ => {}
        }
    }

    /// 4-step mode: quarter/half frame events, IRQ on step 4
    fn clock_frame_counter_4step(&mut self) {
        match self.frame_cycle {
            FRAME_STEP_1 | FRAME_STEP_3 => self.clock_quarter_frame(),
            FRAME_STEP_2 => self.clock_half_and_quarter_frame(),
            FRAME_STEP_4 => self.reset_frame_cycle_with_irq(),
            _ => {}
        }
    }

    /// Clock both quarter-frame and half-frame units
    fn clock_half_and_quarter_frame(&mut self) {
        self.clock_quarter_frame();
        self.clock_half_frame();
    }

    /// Reset frame cycle counter (end of 5-step sequence)
    fn reset_frame_cycle(&mut self) {
        self.clock_quarter_frame();
        self.clock_half_frame();
        self.frame_cycle = 0;
    }

    /// Reset frame cycle counter and fire IRQ if not inhibited (end of 4-step sequence)
    fn reset_frame_cycle_with_irq(&mut self) {
        self.clock_quarter_frame();
        self.clock_half_frame();
        if !self.frame_irq_inhibit {
            self.frame_irq_pending = true;
        }
        self.frame_cycle = 0;
    }

    /// Quarter-frame: clock envelopes and triangle linear counter
    fn clock_quarter_frame(&mut self) {
        self.pulse1.envelope.clock();
        self.pulse2.envelope.clock();
        self.noise.envelope.clock();
        self.triangle.clock_linear();
    }

    /// Half-frame: clock length counters and sweep units
    fn clock_half_frame(&mut self) {
        self.pulse1.length.clock();
        self.pulse2.length.clock();
        self.triangle.length.clock();
        self.noise.length.clock();
        self.pulse1.sweep.clock(&mut self.pulse1.timer_period);
        self.pulse2.sweep.clock(&mut self.pulse2.timer_period);
    }

    /// Mix all channel outputs into a single audio sample
    /// ref: https://www.nesdev.org/wiki/APU_Mixer
    #[inline]
    fn mix(&self) -> f32 {
        let pulse1_level = self.pulse1.output() as usize;
        let pulse2_level = self.pulse2.output() as usize;
        let triangle_level = self.triangle.output() as usize;
        let noise_level = self.noise.output() as usize;
        let dmc_level = self.dmc.output() as usize;

        let pulse_out = self.pulse_table[pulse1_level + pulse2_level];
        // Mixer coefficients from APU reference: 3*triangle + 2*noise + dmc
        let tnd_index = (3 * triangle_level + 2 * noise_level + dmc_level).min(TND_TABLE_SIZE - 1);
        let tnd_out = self.tnd_table[tnd_index];

        pulse_out + tnd_out
    }

    /// Take the audio buffer (returns samples and clears the internal buffer)
    pub fn take_samples(&mut self) -> Vec<f32> {
        std::mem::take(&mut self.buffer)
    }

    /// Check if DMC needs a sample byte from CPU memory
    pub fn dmc_needs_sample(&self) -> bool {
        self.dmc.enabled && self.dmc.sample_buffer.is_none() && self.dmc.bytes_remaining > 0
    }

    /// Provide a sample byte to DMC from CPU memory
    pub fn dmc_load_sample(&mut self, value: NesByte) {
        self.dmc.load_sample(value);
    }

    /// Get the current DMC sample address (for CPU to read from)
    pub fn dmc_sample_address(&self) -> u16 {
        self.dmc.current_address
    }

    /// Check if any APU IRQ is pending
    pub fn irq_pending(&self) -> bool {
        self.frame_irq_pending || self.dmc.irq_pending
    }
}
