use super::bus::{Bus, CpuBus};
use super::common::{NesAddr, NesByte};

// reference: https://www.nesdev.org/wiki/CPU_registers
pub struct Cpu {
    /// Program counter
    pc: NesAddr,
    /// Stack pointer
    sp: NesByte,
    /// Accumulator
    a: NesByte,
    /// Index register X
    x: NesByte,
    /// Index register Y
    y: NesByte,
    /// Status register (NV-BDIZC)
    p: NesByte,

    cycles: u32,
    skip_cycles: u32,

    /// The CPU bus (memory map)
    pub bus: CpuBus,
}

// status mask: https://www.nesdev.org/wiki/Status_flags
// C: Carry
const CARRY: NesByte = 0x1;
// Z: Zero
const ZERO: NesByte = 0x1 << 1;
// I: Interrupt Disable
const INTERRUPT_DISABLE: NesByte = 0x1 << 2;
// D: Decimal
const DECIMAL: NesByte = 0x1 << 3;
const BREAK: NesByte = 0x1 << 4;
const UNUSED: NesByte = 0x1 << 5;
// V: Overflow
const OVERFLOW: NesByte = 0x1 << 6;
// N: Negative
const NEGATIVE: NesByte = 0x1 << 7;
// Bit masks used in instruction operands
const SIGN_BIT: NesAddr = 0x80;
const SIGN_BIT_BYTE: NesByte = 0x80;
const OVERFLOW_BIT: NesByte = 0x40;
const CARRY_OUT_BIT: NesAddr = 0x100;
const STACK_BASE: NesAddr = 0x0100;
const PAGE_MASK: NesAddr = 0xff00;

// ref: https://www.nesdev.org/wiki/CPU_unofficial_opcodes
// implied op
const OP_BRK: NesByte = 0x00;
const OP_PHP: NesByte = 0x08;
const OP_CLC: NesByte = 0x18;
const OP_PLP: NesByte = 0x28;
const OP_SEC: NesByte = 0x38;
const OP_RTI: NesByte = 0x40;
const OP_PHA: NesByte = 0x48;
const OP_CLI: NesByte = 0x58;
const OP_RTS: NesByte = 0x60;
const OP_PLA: NesByte = 0x68;
const OP_SEI: NesByte = 0x78;
const OP_DEY: NesByte = 0x88;
const OP_TXA: NesByte = 0x8A;
const OP_TYA: NesByte = 0x98;
const OP_TXS: NesByte = 0x9A;
const OP_TAY: NesByte = 0xA8;
const OP_TAX: NesByte = 0xAA;
const OP_CLV: NesByte = 0xB8;
const OP_TSX: NesByte = 0xBA;
const OP_INY: NesByte = 0xC8;
const OP_DEX: NesByte = 0xCA;
const OP_CLD: NesByte = 0xD8;
const OP_INX: NesByte = 0xE8;
const OP_NOP: NesByte = 0xEA;
const OP_SED: NesByte = 0xF8;

// jump op
const OP_JSR: NesByte = 0x20;
const OP_JMP: NesByte = 0x4C;
const OP_JMPI: NesByte = 0x6C; // JMP indirect

// branch op
const BRANCH_INSTRUCTION_MASK: NesByte = 0x1f;
const BRANCH_INSTRUCTION_MASK_RESULT: NesByte = 0x10;
const BRANCH_CONDITION_MASK: NesByte = 0x20;
const BRANCH_ON_FLAG_SHIFT: u8 = 6;

// Flag indices for branch decoding (opcode >> 6)
const FLAG_NEGATIVE: NesByte = 0;
const FLAG_OVERFLOW: NesByte = 1;
const FLAG_CARRY: NesByte = 2;
const FLAG_ZERO: NesByte = 3;

// Instruction decoding masks
// ref: https://www.nesdev.org/wiki/CPU_unofficial_opcodes
//
// 6502 opcodes are encoded as: aaabbbcc
//   cc  = instruction mode (type 0, 1, or 2)
//   bbb = addressing mode
//   aaa = operation
const INSTRUCTION_MODE_MASK: NesByte = 0x03;
const OPERATION_MASK: NesByte = 0xE0;
const OPERATION_SHIFT: u8 = 5;
const ADDRESS_MODE_MASK: NesByte = 0x1C;
const ADDRESS_MODE_SHIFT: u8 = 2;

// Addressing modes for type 1 instructions (cc=01)
const M1_INDEXED_INDIRECT_X: NesByte = 0;
const M1_ZERO_PAGE: NesByte = 1;
const M1_IMMEDIATE: NesByte = 2;
const M1_ABSOLUTE: NesByte = 3;
const M1_INDIRECT_Y: NesByte = 4;
const M1_INDEXED_X: NesByte = 5;
const M1_ABSOLUTE_Y: NesByte = 6;
const M1_ABSOLUTE_X: NesByte = 7;

// Addressing modes for type 0/2 instructions (cc=00/cc=10)
const M2_IMMEDIATE: NesByte = 0;
const M2_ZERO_PAGE: NesByte = 1;
const M2_ACCUMULATOR: NesByte = 2;
const M2_ABSOLUTE: NesByte = 3;
const M2_INDEXED: NesByte = 5;
const M2_ABSOLUTE_INDEXED: NesByte = 7;

// Operations for type 1 (cc=01)
const OP1_ORA: NesByte = 0;
const OP1_AND: NesByte = 1;
const OP1_EOR: NesByte = 2;
const OP1_ADC: NesByte = 3;
const OP1_STA: NesByte = 4;
const OP1_LDA: NesByte = 5;
const OP1_CMP: NesByte = 6;
const OP1_SBC: NesByte = 7;

// Operations for type 2 (cc=10)
const OP2_ASL: NesByte = 0;
const OP2_ROL: NesByte = 1;
const OP2_LSR: NesByte = 2;
const OP2_ROR: NesByte = 3;
const OP2_STX: NesByte = 4;
const OP2_LDX: NesByte = 5;
const OP2_DEC: NesByte = 6;
const OP2_INC: NesByte = 7;

// Operations for type 0 (cc=00)
const OP0_BIT: NesByte = 1;
const OP0_STY: NesByte = 4;
const OP0_LDY: NesByte = 5;
const OP0_CPY: NesByte = 6;
const OP0_CPX: NesByte = 7;

// interrupt vector
const NMI_VECTOR: NesAddr = 0xFFFA;
const RESET_VECTOR: NesAddr = 0xFFFC;
const IRQ_VECTOR: NesAddr = 0xFFFE;
const INTERRUPT_CYCLES: u32 = 7;

#[allow(clippy::upper_case_acronyms)]
enum InterruptType {
    IRQ,   // hardware IRQ instruction
    BRK,   // hardware BRK instruction
    NMI,   // PPU vertical blank
    RESET, // Power on/off
}

// ref: https://www.nesdev.org/wiki/6502_cycle_times
#[rustfmt::skip]
const OPERATION_CYCLES: [NesByte; 256] = [
    0, 6, 0, 0, 0, 3, 5, 0, 3, 2, 2, 0, 0, 4, 6, 0,
    2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
    6, 6, 0, 0, 3, 3, 5, 0, 4, 2, 2, 0, 4, 4, 6, 0,
    2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
    6, 6, 0, 0, 0, 3, 5, 0, 3, 2, 2, 0, 3, 4, 6, 0,
    2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
    6, 6, 0, 0, 0, 3, 5, 0, 4, 2, 2, 0, 5, 4, 6, 0,
    2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
    0, 6, 0, 0, 3, 3, 3, 0, 2, 0, 2, 0, 4, 4, 4, 0,
    2, 6, 0, 0, 4, 4, 4, 0, 2, 5, 2, 0, 0, 5, 0, 0,
    2, 6, 2, 0, 3, 3, 3, 0, 2, 2, 2, 0, 4, 4, 4, 0,
    2, 5, 0, 0, 4, 4, 4, 0, 2, 4, 2, 0, 4, 4, 4, 0,
    2, 6, 0, 0, 3, 3, 5, 0, 2, 2, 2, 0, 4, 4, 6, 0,
    2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
    2, 6, 0, 0, 3, 3, 5, 0, 2, 2, 2, 2, 4, 4, 6, 0,
    2, 5, 0, 0, 0, 4, 6, 0, 2, 4, 0, 0, 0, 4, 7, 0,
];

// Lifecycle & public API
impl Cpu {
    pub fn new(bus: CpuBus) -> Self {
        Self {
            pc: 0,
            sp: 0xFD,
            a: 0,
            x: 0,
            y: 0,
            p: INTERRUPT_DISABLE | UNUSED,
            cycles: 0,
            skip_cycles: 0,
            bus,
        }
    }

    /// Reset the CPU: initialize registers and load PC from RESET vector
    pub fn reset(&mut self) {
        self.skip_cycles = 0;
        self.cycles = 0;
        self.a = 0;
        self.x = 0;
        self.y = 0;
        self.p = INTERRUPT_DISABLE | UNUSED; // 0b00110100
        self.sp = 0xFD;
        self.pc = self.read_addr(RESET_VECTOR);
    }

    /// Trigger a non-maskable interrupt (called by PPU on vblank)
    pub fn nmi(&mut self) {
        self.interrupt(InterruptType::NMI);
    }

    /// Trigger a hardware IRQ interrupt
    pub fn irq(&mut self) {
        self.interrupt(InterruptType::IRQ);
    }

    /// Execute one CPU instruction, returning the number of cycles consumed
    /// (includes extra cycles from page crossings, branches, and interrupts)
    #[inline]
    pub fn step(&mut self) -> u32 {
        let op = self.bus.read(self.pc);
        self.pc = self.pc.wrapping_add(1);

        let cycle_length = OPERATION_CYCLES[op as usize] as u32;

        // Dispatch chain: try each instruction category in order.
        // Each returns true if it handled the opcode, short-circuiting the rest.
        if !(self.implied(op)
            || self.branch(op)
            || self.jump(op)
            || self.type0(op)
            || self.type1(op)
            || self.type2(op))
        {
            // Unknown/unofficial opcode — treat as NOP
        }

        // Include extra cycles from page crossings, branches, interrupts, and DMA
        let total = cycle_length + self.skip_cycles + self.bus.dma_cycles;
        self.skip_cycles = 0;
        self.bus.dma_cycles = 0;
        self.cycles += total;
        total
    }

    /// Return the total number of cycles executed so far
    pub fn cycles(&self) -> u32 {
        self.cycles
    }
}

// Flags & utilities
impl Cpu {
    fn set_zero(&mut self, value: NesByte) {
        if value == 0 {
            self.p |= ZERO;
        } else {
            self.p &= !ZERO;
        }
    }

    fn set_negative(&mut self, value: NesByte) {
        if value & NEGATIVE != 0 {
            self.p |= NEGATIVE
        } else {
            self.p &= !NEGATIVE
        }
    }

    /// Set both Zero and Negative flags from a byte value
    fn set_zn(&mut self, value: NesByte) {
        self.set_zero(value);
        self.set_negative(value);
    }

    /// Set Zero and Negative flags from a 16-bit value (uses low byte)
    fn set_zn_wide(&mut self, value: NesAddr) {
        self.set_zn(value as NesByte);
    }

    /// Set or clear the Carry flag
    fn set_carry(&mut self, condition: bool) {
        if condition {
            self.p |= CARRY;
        } else {
            self.p &= !CARRY;
        }
    }

    /// Set or clear the Overflow flag
    fn set_overflow(&mut self, condition: bool) {
        if condition {
            self.p |= OVERFLOW;
        } else {
            self.p &= !OVERFLOW;
        }
    }

    // Reads a 16-bit little-endian address from two consecutive bytes
    fn read_addr(&mut self, addr: NesAddr) -> NesAddr {
        let lo = self.bus.read(addr) as NesAddr;
        let hi = self.bus.read(addr + 1) as NesAddr;
        lo | (hi << 8)
    }

    // ref: https://www.nesdev.org/wiki/Stack
    fn push_stack(&mut self, value: NesByte) {
        self.bus.write(STACK_BASE | self.sp as NesAddr, value);
        self.sp = self.sp.wrapping_sub(1);
    }

    fn pop_stack(&mut self) -> NesByte {
        self.sp = self.sp.wrapping_add(1);
        self.bus.read(STACK_BASE | self.sp as NesAddr)
    }

    // ref: https://www.nesdev.org/wiki/CPU_addressing_modes
    fn set_page_crossed(&mut self, a: NesAddr, b: NesAddr) {
        if (a & PAGE_MASK) != (b & PAGE_MASK) {
            self.skip_cycles += 1;
        }
    }
}

// Addressing modes
// ref: https://www.nesdev.org/wiki/CPU_addressing_modes
impl Cpu {
    /// Immediate: operand is next byte
    fn addr_immediate(&mut self) -> NesAddr {
        let loc = self.pc;
        self.pc = self.pc.wrapping_add(1);
        loc
    }

    /// Zero Page: operand address is a single byte (page 0)
    fn addr_zero_page(&mut self) -> NesAddr {
        let loc = self.bus.read(self.pc) as NesAddr;
        self.pc = self.pc.wrapping_add(1);
        loc
    }

    /// Absolute: operand address is a full 16-bit address
    fn addr_absolute(&mut self) -> NesAddr {
        let loc = self.read_addr(self.pc);
        self.pc = self.pc.wrapping_add(2);
        loc
    }

    /// Zero Page,X: zero page address + X, wraps around in page 0
    fn addr_zero_page_x(&mut self) -> NesAddr {
        let base = self.bus.read(self.pc);
        let loc = base.wrapping_add(self.x) as NesAddr & 0xFF;
        self.pc = self.pc.wrapping_add(1);
        loc
    }

    /// Zero Page,Y: zero page address + Y, wraps around in page 0
    fn addr_zero_page_y(&mut self) -> NesAddr {
        let base = self.bus.read(self.pc);
        let loc = base.wrapping_add(self.y) as NesAddr & 0xFF;
        self.pc = self.pc.wrapping_add(1);
        loc
    }

    /// Absolute,X: absolute address + X, checks page crossing
    fn addr_absolute_x(&mut self) -> NesAddr {
        let loc = self.read_addr(self.pc);
        self.pc = self.pc.wrapping_add(2);
        self.set_page_crossed(loc, loc.wrapping_add(self.x as NesAddr));
        loc.wrapping_add(self.x as NesAddr)
    }

    /// Absolute,X without page crossing check (used by STA)
    fn addr_absolute_x_no_check(&mut self) -> NesAddr {
        let loc = self.read_addr(self.pc);
        self.pc = self.pc.wrapping_add(2);
        loc.wrapping_add(self.x as NesAddr)
    }

    /// Absolute,Y: absolute address + Y, checks page crossing
    fn addr_absolute_y(&mut self) -> NesAddr {
        let loc = self.read_addr(self.pc);
        self.pc = self.pc.wrapping_add(2);
        self.set_page_crossed(loc, loc.wrapping_add(self.y as NesAddr));
        loc.wrapping_add(self.y as NesAddr)
    }

    /// Absolute,Y without page crossing check (used by STA)
    fn addr_absolute_y_no_check(&mut self) -> NesAddr {
        let loc = self.read_addr(self.pc);
        self.pc = self.pc.wrapping_add(2);
        loc.wrapping_add(self.y as NesAddr)
    }

    /// (Indirect,X): zero page address + X -> 16-bit pointer
    fn addr_indexed_indirect_x(&mut self) -> NesAddr {
        let zero_addr = self.bus.read(self.pc).wrapping_add(self.x);
        self.pc = self.pc.wrapping_add(1);
        let lo = self.bus.read(zero_addr as NesAddr) as NesAddr;
        let hi = self.bus.read(zero_addr.wrapping_add(1) as NesAddr) as NesAddr;
        lo | (hi << 8)
    }

    /// (Indirect),Y: zero page -> 16-bit pointer + Y, checks page crossing
    fn addr_indirect_y(&mut self) -> NesAddr {
        let zero_addr = self.bus.read(self.pc);
        self.pc = self.pc.wrapping_add(1);
        let lo = self.bus.read(zero_addr as NesAddr) as NesAddr;
        let hi = self.bus.read(zero_addr.wrapping_add(1) as NesAddr) as NesAddr;
        let loc = lo | (hi << 8);
        self.set_page_crossed(loc, loc.wrapping_add(self.y as NesAddr));
        loc.wrapping_add(self.y as NesAddr)
    }

    /// (Indirect),Y without page crossing check (used by STA)
    fn addr_indirect_y_no_check(&mut self) -> NesAddr {
        let zero_addr = self.bus.read(self.pc);
        self.pc = self.pc.wrapping_add(1);
        let lo = self.bus.read(zero_addr as NesAddr) as NesAddr;
        let hi = self.bus.read(zero_addr.wrapping_add(1) as NesAddr) as NesAddr;
        let loc = lo | (hi << 8);
        loc.wrapping_add(self.y as NesAddr)
    }
}

// Instruction dispatch
// ref: https://www.masswerk.at/6502/6502_instruction_set.html
impl Cpu {
    fn implied(&mut self, op: NesByte) -> bool {
        match op {
            OP_BRK => self.brk(),
            OP_PHP => self.php(),
            OP_CLC => self.clc(),
            OP_PLP => self.plp(),
            OP_SEC => self.sec(),
            OP_RTI => self.rti(),
            OP_PHA => self.pha(),
            OP_CLI => self.cli(),
            OP_RTS => self.rts(),
            OP_PLA => self.pla(),
            OP_SEI => self.sei(),
            OP_DEY => self.dey(),
            OP_TXA => self.txa(),
            OP_TYA => self.tya(),
            OP_TXS => self.txs(),
            OP_TAY => self.tay(),
            OP_TAX => self.tax(),
            OP_CLV => self.clv(),
            OP_TSX => self.tsx(),
            OP_INY => self.iny(),
            OP_DEX => self.dex(),
            OP_CLD => self.cld(),
            OP_INX => self.inx(),
            OP_NOP => self.nop(),
            OP_SED => self.sed(),
            _ => false,
        }
    }

    // ref: https://www.nesdev.org/wiki/CPU_unofficial_opcodes
    //
    // All branch instructions share a common encoding:
    //   bit 7-6: flag index (N=0, V=1, C=2, Z=3)
    //   bit 5:   condition (0 = flag clear, 1 = flag set)
    //   bit 4-0: 0x10 identifies a branch instruction
    //
    // Branch opcodes:
    //   BPL ($10): branch if N=0    BMI ($30): branch if N=1
    //   BVC ($50): branch if V=0    BVS ($70): branch if V=1
    //   BCC ($90): branch if C=0    BCS ($B0): branch if C=1
    //   BNE ($D0): branch if Z=0    BEQ ($F0): branch if Z=1
    fn branch(&mut self, op: NesByte) -> bool {
        if (op & BRANCH_INSTRUCTION_MASK) != BRANCH_INSTRUCTION_MASK_RESULT {
            return false;
        }

        // Determine whether we want the flag set (e.g., BMI, BCS) or clear (e.g., BPL, BCC)
        let condition = op & BRANCH_CONDITION_MASK != 0;

        // Read the flag bit selected by opcode bits 7-6
        let flag_set = match op >> BRANCH_ON_FLAG_SHIFT {
            FLAG_NEGATIVE => self.p & NEGATIVE != 0,
            FLAG_OVERFLOW => self.p & OVERFLOW != 0,
            FLAG_CARRY => self.p & CARRY != 0,
            FLAG_ZERO => self.p & ZERO != 0,
            _ => return false,
        };

        // XNOR: branch if condition matches flag state (equivalent to condition == flag_set)
        let should_branch = !(condition ^ flag_set);

        if should_branch {
            let offset = self.bus.read(self.pc) as i8;
            self.pc = self.pc.wrapping_add(1);
            self.skip_cycles += 1;
            let new_pc = self.pc.wrapping_add(offset as u16);
            self.set_page_crossed(self.pc, new_pc);
            self.pc = new_pc;
        } else {
            self.pc = self.pc.wrapping_add(1);
        }

        true
    }

    // Jump instructions: JSR, JMP, JMP indirect
    fn jump(&mut self, op: NesByte) -> bool {
        match op {
            OP_JSR => self.jsr(),
            OP_JMP => self.jmp(),
            OP_JMPI => self.jmp_indirect(),
            _ => return false,
        }
        true
    }

    // Type 0 instructions (cc=00): BIT, STY, LDY, CPY, CPX
    // Addressing modes: immediate, zero page, absolute, zero page X, absolute X
    fn type0(&mut self, op: NesByte) -> bool {
        if (op & INSTRUCTION_MODE_MASK) != 0x00 {
            return false;
        }

        let addr_mode = (op & ADDRESS_MODE_MASK) >> ADDRESS_MODE_SHIFT;
        let location = match addr_mode {
            M2_IMMEDIATE => self.addr_immediate(),
            M2_ZERO_PAGE => self.addr_zero_page(),
            M2_ABSOLUTE => self.addr_absolute(),
            M2_INDEXED => self.addr_zero_page_x(),
            M2_ABSOLUTE_INDEXED => self.addr_absolute_x(),
            _ => return false,
        };

        let operation = (op & OPERATION_MASK) >> OPERATION_SHIFT;
        match operation {
            OP0_BIT => self.op_bit(location),
            OP0_STY => self.op_sty(location),
            OP0_LDY => self.op_ldy(location),
            OP0_CPY => self.op_cpy(location),
            OP0_CPX => self.op_cpx(location),
            _ => return false,
        }

        true
    }

    // Type 1 instructions (cc=01): ORA, AND, EOR, ADC, STA, LDA, CMP, SBC
    // Addressing modes: (ind,X), zpg, #imm, abs, (ind),Y, zpg X, abs Y, abs X
    fn type1(&mut self, op: NesByte) -> bool {
        if (op & INSTRUCTION_MODE_MASK) != 0x01 {
            return false;
        }

        let operation = (op & OPERATION_MASK) >> OPERATION_SHIFT;
        let addr_mode = (op & ADDRESS_MODE_MASK) >> ADDRESS_MODE_SHIFT;
        let is_sta = operation == OP1_STA;

        // STA skips page crossing checks (no extra cycle penalty)
        let location = match addr_mode {
            M1_INDEXED_INDIRECT_X => self.addr_indexed_indirect_x(),
            M1_ZERO_PAGE => self.addr_zero_page(),
            M1_IMMEDIATE => self.addr_immediate(),
            M1_ABSOLUTE => self.addr_absolute(),
            M1_INDIRECT_Y if is_sta => self.addr_indirect_y_no_check(),
            M1_INDIRECT_Y => self.addr_indirect_y(),
            M1_INDEXED_X => self.addr_zero_page_x(),
            M1_ABSOLUTE_Y if is_sta => self.addr_absolute_y_no_check(),
            M1_ABSOLUTE_Y => self.addr_absolute_y(),
            M1_ABSOLUTE_X if is_sta => self.addr_absolute_x_no_check(),
            M1_ABSOLUTE_X => self.addr_absolute_x(),
            _ => return false,
        };

        match operation {
            OP1_ORA => self.op_ora(location),
            OP1_AND => self.op_and(location),
            OP1_EOR => self.op_eor(location),
            OP1_ADC => self.op_adc(location),
            OP1_STA => self.op_sta(location),
            OP1_LDA => self.op_lda(location),
            OP1_CMP => self.op_cmp(location),
            OP1_SBC => self.op_sbc(location),
            _ => return false,
        }

        true
    }

    // Type 2 instructions (cc=10): ASL, ROL, LSR, ROR, STX, LDX, DEC, INC
    // Addressing modes: #imm, zpg, acc, abs, zpg X/Y, abs X/Y
    fn type2(&mut self, op: NesByte) -> bool {
        if (op & INSTRUCTION_MODE_MASK) != 0x02 {
            return false;
        }

        let operation = (op & OPERATION_MASK) >> OPERATION_SHIFT;
        let addr_mode = (op & ADDRESS_MODE_MASK) >> ADDRESS_MODE_SHIFT;
        // STX/LDX use Y index for indexed modes, all others use X
        let uses_y = operation == OP2_STX || operation == OP2_LDX;

        let is_rmw = matches!(
            operation,
            OP2_ASL | OP2_ROL | OP2_LSR | OP2_ROR | OP2_DEC | OP2_INC
        );

        let location = match addr_mode {
            M2_IMMEDIATE => self.addr_immediate(),
            M2_ZERO_PAGE => self.addr_zero_page(),
            M2_ACCUMULATOR => 0, // not used as address
            M2_ABSOLUTE => self.addr_absolute(),
            M2_INDEXED if uses_y => self.addr_zero_page_y(),
            M2_INDEXED => self.addr_zero_page_x(),
            M2_ABSOLUTE_INDEXED if uses_y => self.addr_absolute_y(),
            // RMW instructions (ASL/ROL/LSR/ROR/DEC/INC) always take 7 cycles
            // with no page-crossing penalty
            M2_ABSOLUTE_INDEXED if is_rmw => self.addr_absolute_x_no_check(),
            M2_ABSOLUTE_INDEXED => self.addr_absolute_x(),
            _ => return false,
        };

        match operation {
            OP2_ASL => self.op_shift_left(location, addr_mode, false),
            OP2_ROL => self.op_shift_left(location, addr_mode, true),
            OP2_LSR => self.op_shift_right(location, addr_mode, false),
            OP2_ROR => self.op_shift_right(location, addr_mode, true),
            OP2_STX => self.op_stx(location),
            OP2_LDX => self.op_ldx(location),
            OP2_DEC => self.op_dec(location),
            OP2_INC => self.op_inc(location),
            _ => return false,
        }

        true
    }
}

// Implied operations
impl Cpu {
    // BRK = Break — a software interrupt instruction (opcode 0x00).
    fn brk(&mut self) -> bool {
        self.interrupt(InterruptType::BRK);
        true
    }

    // PHP = Push Processor status — pushes the status register (P) onto the stack.
    // ref: https://www.nesdev.org/wiki/Status_flags#The_B_flag
    fn php(&mut self) -> bool {
        self.push_stack(self.p | BREAK | UNUSED);
        true
    }

    // CLC = Clear Carry flag.
    fn clc(&mut self) -> bool {
        self.p &= !CARRY;
        true
    }

    // PLP = Pull Processor status — pops the status register (P) from the stack.
    // ref: https://www.nesdev.org/wiki/Status_flags#The_B_flag
    fn plp(&mut self) -> bool {
        self.p = self.pop_stack();
        // BREAK and UNUSED bits are ignored when pulling
        self.p &= !(BREAK | UNUSED);
        self.p |= UNUSED; // bit 5 is always set
        true
    }

    // SEC = Set Carry flag — sets bit 0 of the status register to 1.
    fn sec(&mut self) -> bool {
        self.p |= CARRY;
        true
    }

    // RTI = Return from Interrupt
    // ref: https://www.nesdev.org/wiki/RTI
    fn rti(&mut self) -> bool {
        // pull status register from stack
        self.p = self.pop_stack();
        self.p &= !BREAK;
        self.p |= UNUSED;
        // pull PC from stack (low byte first, then high byte)
        let lo = self.pop_stack() as NesAddr;
        let hi = self.pop_stack() as NesAddr;
        self.pc = lo | (hi << 8);
        true
    }

    // PHA = Push Accumulator — pushes register A onto the stack.
    fn pha(&mut self) -> bool {
        self.push_stack(self.a);
        true
    }

    // CLI = Clear Interrupt Disable — allows hardware IRQs to fire.
    fn cli(&mut self) -> bool {
        self.p &= !INTERRUPT_DISABLE;
        true
    }

    // RTS = Return from Subroutine — pulls the return address from the stack and jumps to it +1.
    fn rts(&mut self) -> bool {
        let lo = self.pop_stack() as NesAddr;
        let hi = self.pop_stack() as NesAddr;
        self.pc = (lo | (hi << 8)) + 1;
        true
    }

    // PLA = Pull Accumulator — pops a value from the stack into register A. Sets Zero and Negative flags.
    fn pla(&mut self) -> bool {
        self.a = self.pop_stack();
        self.set_zero(self.a);
        self.set_negative(self.a);
        true
    }

    // SEI = Set Interrupt Disable — prevents hardware IRQs from firing. The opposite of CLI.
    fn sei(&mut self) -> bool {
        self.p |= INTERRUPT_DISABLE;
        true
    }

    // DEY = Decrement Y register — subtracts 1 from register Y. Sets Zero and Negative flags.
    fn dey(&mut self) -> bool {
        self.y = self.y.wrapping_sub(1);
        self.set_zero(self.y);
        self.set_negative(self.y);
        true
    }

    // TXA = Transfer X to Accumulator — copies register X into register A. Sets Zero and Negative flags.
    fn txa(&mut self) -> bool {
        self.a = self.x;
        self.set_zero(self.a);
        self.set_negative(self.a);
        true
    }

    // TYA = Transfer Y to Accumulator — copies register Y into register A. Sets Zero and Negative flags.
    fn tya(&mut self) -> bool {
        self.a = self.y;
        self.set_zero(self.a);
        self.set_negative(self.a);
        true
    }

    // TXS = Transfer X to Stack pointer — copies register X into the stack pointer. No flags affected.
    fn txs(&mut self) -> bool {
        self.sp = self.x;
        true
    }

    // TAY = Transfer Accumulator to Y — copies register A into register Y. Sets Zero and Negative flags.
    fn tay(&mut self) -> bool {
        self.y = self.a;
        self.set_zero(self.y);
        self.set_negative(self.y);
        true
    }

    // TAX = Transfer Accumulator to X — copies register A into register X. Sets Zero and Negative flags.
    fn tax(&mut self) -> bool {
        self.x = self.a;
        self.set_zero(self.x);
        self.set_negative(self.x);
        true
    }

    // CLV = Clear Overflow flag.
    fn clv(&mut self) -> bool {
        self.p &= !OVERFLOW;
        true
    }

    // TSX = Transfer Stack pointer to X. Sets Zero and Negative flags.
    fn tsx(&mut self) -> bool {
        self.x = self.sp;
        self.set_zero(self.x);
        self.set_negative(self.x);
        true
    }

    // INY = Increment Y register. Sets Zero and Negative flags.
    fn iny(&mut self) -> bool {
        self.y = self.y.wrapping_add(1);
        self.set_zero(self.y);
        self.set_negative(self.y);
        true
    }

    // DEX = Decrement X register. Sets Zero and Negative flags.
    fn dex(&mut self) -> bool {
        self.x = self.x.wrapping_sub(1);
        self.set_zero(self.x);
        self.set_negative(self.x);
        true
    }

    // CLD = Clear Decimal flag.
    fn cld(&mut self) -> bool {
        self.p &= !DECIMAL;
        true
    }

    // INX = Increment X register. Sets Zero and Negative flags.
    fn inx(&mut self) -> bool {
        self.x = self.x.wrapping_add(1);
        self.set_zero(self.x);
        self.set_negative(self.x);
        true
    }

    // NOP = No Operation. Does nothing.
    fn nop(&mut self) -> bool {
        true
    }

    // SED = Set Decimal flag.
    fn sed(&mut self) -> bool {
        self.p |= DECIMAL;
        true
    }
}

// Jump operations
impl Cpu {
    // JSR = Jump to Subroutine — pushes return address - 1, then jumps to target
    // ref: https://www.nesdev.org/wiki/JSR
    fn jsr(&mut self) {
        let target = self.read_addr(self.pc);
        // Push return address - 1 (6502 convention: JSR pushes PC pointing to last byte of JSR)
        let return_addr = self.pc.wrapping_add(1);
        self.push_stack((return_addr >> 8) as NesByte);
        self.push_stack(return_addr as NesByte);
        self.pc = target;
    }

    // JMP = Jump — unconditional jump to absolute address
    fn jmp(&mut self) {
        self.pc = self.read_addr(self.pc);
    }

    // JMP indirect = Jump through a pointer
    // Note: 6502 bug — if the pointer is at $xxFF, the high byte is fetched
    // from $xx00 instead of $xx00+$0100 (page boundary wrapping bug)
    // ref: https://www.nesdev.org/wiki/Errata#CPU
    fn jmp_indirect(&mut self) {
        let pointer = self.read_addr(self.pc);
        let lo = self.bus.read(pointer) as NesAddr;
        // Page boundary bug: high byte wraps within the same page
        let hi_addr = (pointer & 0xFF00) | ((pointer + 1) & 0x00FF);
        let hi = self.bus.read(hi_addr) as NesAddr;
        self.pc = lo | (hi << 8);
    }
}

// Type 0 operations (cc=00): BIT, STY, LDY, CPY, CPX
impl Cpu {
    /// BIT — test bits in memory with accumulator
    fn op_bit(&mut self, location: NesAddr) {
        let operand = self.bus.read(location);
        self.set_zero(self.a & operand);
        self.set_overflow(operand & OVERFLOW_BIT != 0);
        self.set_negative(operand);
    }

    /// STY — store Y register in memory
    fn op_sty(&mut self, location: NesAddr) {
        self.bus.write(location, self.y);
    }

    /// LDY — load Y register from memory
    fn op_ldy(&mut self, location: NesAddr) {
        self.y = self.bus.read(location);
        self.set_zn(self.y);
    }

    /// CPY — compare Y register with memory
    fn op_cpy(&mut self, location: NesAddr) {
        let diff = (self.y as NesAddr).wrapping_sub(self.bus.read(location) as NesAddr);
        self.set_carry(diff & 0x100 == 0);
        self.set_zn_wide(diff);
    }

    /// CPX — compare X register with memory
    fn op_cpx(&mut self, location: NesAddr) {
        let diff = (self.x as NesAddr).wrapping_sub(self.bus.read(location) as NesAddr);
        self.set_carry(diff & 0x100 == 0);
        self.set_zn_wide(diff);
    }
}

// Type 1 operations (cc=01): ORA, AND, EOR, ADC, STA, LDA, CMP, SBC
impl Cpu {
    /// ORA — OR memory with accumulator
    fn op_ora(&mut self, location: NesAddr) {
        self.a |= self.bus.read(location);
        self.set_zn(self.a);
    }

    /// AND — AND memory with accumulator
    fn op_and(&mut self, location: NesAddr) {
        self.a &= self.bus.read(location);
        self.set_zn(self.a);
    }

    /// EOR — exclusive OR memory with accumulator
    fn op_eor(&mut self, location: NesAddr) {
        self.a ^= self.bus.read(location);
        self.set_zn(self.a);
    }

    /// ADC — add memory to accumulator with carry
    fn op_adc(&mut self, location: NesAddr) {
        let operand = self.bus.read(location);
        let carry = if self.p & CARRY != 0 { 1u16 } else { 0u16 };
        let sum = self.a as NesAddr + operand as NesAddr + carry;
        self.set_carry(sum & CARRY_OUT_BIT != 0);
        // Overflow: sign of result differs from BOTH operands
        let a_sign_changed = (self.a as NesAddr ^ sum) & SIGN_BIT;
        let operand_sign_changed = (operand as NesAddr ^ sum) & SIGN_BIT;
        self.set_overflow(a_sign_changed & operand_sign_changed != 0);
        self.a = sum as NesByte;
        self.set_zn(self.a);
    }

    /// STA — store accumulator in memory
    fn op_sta(&mut self, location: NesAddr) {
        self.bus.write(location, self.a);
    }

    /// LDA — load accumulator from memory
    fn op_lda(&mut self, location: NesAddr) {
        self.a = self.bus.read(location);
        self.set_zn(self.a);
    }

    /// CMP — compare accumulator with memory
    fn op_cmp(&mut self, location: NesAddr) {
        let diff = (self.a as NesAddr).wrapping_sub(self.bus.read(location) as NesAddr);
        self.set_carry(diff & 0x100 == 0);
        self.set_zn_wide(diff);
    }

    /// SBC — subtract memory from accumulator with borrow
    fn op_sbc(&mut self, location: NesAddr) {
        let subtrahend = self.bus.read(location) as NesAddr;
        let borrow = if self.p & CARRY != 0 { 0u16 } else { 1u16 };
        let diff = (self.a as NesAddr)
            .wrapping_sub(subtrahend)
            .wrapping_sub(borrow);
        // Carry clear means borrow occurred (bit 8 set)
        self.set_carry(diff & CARRY_OUT_BIT == 0);
        // Overflow: sign changed unexpectedly
        let a_sign_changed = (self.a as NesAddr ^ diff) & SIGN_BIT;
        let operand_sign_changed = (!subtrahend ^ diff) & SIGN_BIT;
        self.set_overflow(a_sign_changed & operand_sign_changed != 0);
        self.a = diff as NesByte;
        self.set_zn_wide(diff);
    }
}

// Type 2 operations (cc=10): ASL, ROL, LSR, ROR, STX, LDX, DEC, INC
impl Cpu {
    /// ASL/ROL — shift left (rotate = feed old carry into bit 0)
    fn op_shift_left(&mut self, location: NesAddr, addr_mode: NesByte, rotate: bool) {
        let prev_carry = if rotate && self.p & CARRY != 0 { 1 } else { 0 };
        if addr_mode == M2_ACCUMULATOR {
            self.set_carry(self.a & SIGN_BIT_BYTE != 0);
            self.a = (self.a << 1) | prev_carry;
            self.set_zn(self.a);
        } else {
            let mut operand = self.bus.read(location);
            self.set_carry(operand & SIGN_BIT_BYTE != 0);
            operand = (operand << 1) | prev_carry;
            self.set_zn(operand);
            self.bus.write(location, operand);
        }
    }

    /// LSR/ROR — shift right (rotate = feed old carry into bit 7)
    fn op_shift_right(&mut self, location: NesAddr, addr_mode: NesByte, rotate: bool) {
        let prev_carry = if rotate && self.p & CARRY != 0 {
            SIGN_BIT_BYTE
        } else {
            0
        };
        if addr_mode == M2_ACCUMULATOR {
            self.set_carry(self.a & 1 != 0);
            self.a = (self.a >> 1) | prev_carry;
            self.set_zn(self.a);
        } else {
            let mut operand = self.bus.read(location);
            self.set_carry(operand & 1 != 0);
            operand = (operand >> 1) | prev_carry;
            self.set_zn(operand);
            self.bus.write(location, operand);
        }
    }

    /// STX — store X register in memory
    fn op_stx(&mut self, location: NesAddr) {
        self.bus.write(location, self.x);
    }

    /// LDX — load X register from memory
    fn op_ldx(&mut self, location: NesAddr) {
        self.x = self.bus.read(location);
        self.set_zn(self.x);
    }

    /// DEC — decrement memory by one
    fn op_dec(&mut self, location: NesAddr) {
        let value = self.bus.read(location).wrapping_sub(1);
        self.set_zn(value);
        self.bus.write(location, value);
    }

    /// INC — increment memory by one
    fn op_inc(&mut self, location: NesAddr) {
        let value = self.bus.read(location).wrapping_add(1);
        self.set_zn(value);
        self.bus.write(location, value);
    }
}

// Interrupt handling
// ref: https://www.nesdev.org/wiki/CPU_interrupts
impl Cpu {
    // Interrupt sequence:
    // 1. Push PC (high byte first) and status register to stack
    // 2. Set Interrupt Disable flag
    // 3. Load PC from interrupt vector
    //
    // Vectors:
    //   NMI:     0xFFFA
    //   RESET:   0xFFFC
    //   IRQ/BRK: 0xFFFE
    fn interrupt(&mut self, typ: InterruptType) {
        let is_irq_disabled = self.p & INTERRUPT_DISABLE != 0;
        if is_irq_disabled && matches!(typ, InterruptType::IRQ) {
            return;
        }

        // BRK increments PC by 1 (6502 quirk)
        if matches!(typ, InterruptType::BRK) {
            self.pc += 1;
        }

        // push PC and status to stack
        self.push_stack((self.pc >> 8) as NesByte);
        self.push_stack(self.pc as NesByte);
        let brk = match typ {
            InterruptType::BRK => BREAK,
            _ => 0,
        };

        self.push_stack(self.p | UNUSED | brk);

        // set interrupt disable flag
        self.p |= INTERRUPT_DISABLE;

        // jump to interrupt vector
        self.pc = match typ {
            InterruptType::IRQ | InterruptType::BRK => self.read_addr(IRQ_VECTOR),
            InterruptType::NMI => self.read_addr(NMI_VECTOR),
            InterruptType::RESET => self.read_addr(RESET_VECTOR),
        };
        self.skip_cycles += INTERRUPT_CYCLES;
    }
}
