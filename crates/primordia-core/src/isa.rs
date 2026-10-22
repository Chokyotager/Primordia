//! PISA-1 instruction set: opcode map, monomer classes, mirror map, default
//! costs and disassembly. Single source of truth for CPU and GPU.

pub mod op {
    pub const NOP: u8 = 0x00;
    pub const DUP: u8 = 0x01;
    pub const DROP: u8 = 0x02;
    pub const SWAP: u8 = 0x03;
    pub const OVER: u8 = 0x04;
    pub const INC: u8 = 0x05;
    pub const DEC: u8 = 0x06;
    pub const ADD: u8 = 0x07;
    pub const SUB: u8 = 0x08;
    pub const SHL4: u8 = 0x09;
    pub const NEG: u8 = 0x0A;
    pub const AND: u8 = 0x0B;
    pub const SKIPNZ: u8 = 0x0C;
    pub const SKIPZ: u8 = 0x0D;
    pub const JREL: u8 = 0x0E;
    pub const JNZ: u8 = 0x0F;
    pub const RD: u8 = 0x10;
    pub const WR: u8 = 0x11;
    pub const CPY: u8 = 0x12;
    pub const H0INC: u8 = 0x13;
    pub const H0DEC: u8 = 0x14;
    pub const H1INC: u8 = 0x15;
    pub const H1DEC: u8 = 0x16;
    pub const SETH0: u8 = 0x17;
    pub const SETH1: u8 = 0x18;
    pub const GETH0: u8 = 0x19;
    pub const GETH1: u8 = 0x1A;
    pub const XH: u8 = 0x1B;
    pub const LEN: u8 = 0x1C;
    pub const GROW: u8 = 0x1D;
    pub const SHRINK: u8 = 0x1E;
    pub const PLEN: u8 = 0x1F;
    pub const YAW_L: u8 = 0x20;
    pub const PITCH_L: u8 = 0x22;
    pub const ROLL_L: u8 = 0x24;
    pub const BONDS_L: u8 = 0x26;
    pub const SENS_L: u8 = 0x28;
    pub const STEP_L: u8 = 0x2A;
    pub const CAT_L: u8 = 0x2C;
    pub const TAKE_L: u8 = 0x2E;
    pub const EMIT_L: u8 = 0x30;
    pub const SIG_L: u8 = 0x32;
    pub const CHEM_L: u8 = 0x34;
    pub const FILL_L0: u8 = 0x36;
    pub const PUSH0: u8 = 0x40;
    pub const MOVE: u8 = 0x60;
    pub const BOND: u8 = 0x61;
    pub const UNBOND: u8 = 0x62;
    pub const SENSE: u8 = 0x63;
    pub const ENRG: u8 = 0x64;
    pub const TEMP: u8 = 0x65;
    pub const SHARE: u8 = 0x66;
    pub const ALLOC: u8 = 0x67;
    pub const DIVIDE: u8 = 0x68;
    pub const DIGEST: u8 = 0x69;
    pub const TUMBLE: u8 = 0x6A;
    pub const FREE: u8 = 0x6B;
    pub const POOL: u8 = 0x6C;
    pub const RAND: u8 = 0x6D;
    pub const SLEEP: u8 = 0x6E;
    pub const HALT: u8 = 0x6F;
}

/// Monomer classes.
pub const CL_L: u8 = 0;
pub const CL_D: u8 = 1;
pub const CL_A: u8 = 2;
pub const CL_NONE: u8 = 3;

#[inline]
pub const fn is_chiral(b: u8) -> bool {
    b >= 0x20 && b <= 0x3F
}

#[inline]
pub const fn mirror(b: u8) -> u8 {
    if is_chiral(b) { b ^ 1 } else { b }
}

#[inline]
pub const fn class(b: u8) -> u8 {
    if b == 0 {
        CL_NONE
    } else if is_chiral(b) {
        b & 1
    } else {
        CL_A
    }
}

/// Hand of a chiral byte: 0 = L, 1 = D.
#[inline]
pub const fn hand(b: u8) -> u8 {
    b & 1
}

/// Default per-opcode cost (energy units). Dynamic parts (TAKE n/4, ALLOC n,
/// DIVIDE n, DIGEST per byte, MOVE into occupied) are added at execution time.
pub fn default_cost(b: u8) -> u32 {
    use op::*;
    match b {
        NOP => 0,
        0x01..=0x0F => 1,
        WR | CPY => 2,
        GROW => 4,
        SHRINK => 2,
        0x10..=0x1F => 1,
        0x20..=0x25 => 1,
        0x26 | 0x27 => 10,
        0x28 | 0x29 => 1,
        0x2A | 0x2B => 20,
        0x2C | 0x2D => 2,
        0x2E | 0x2F => 1,
        0x30 | 0x31 => 3,
        0x32..=0x3F => 1,
        0x40..=0x5F => 1,
        MOVE => 20,
        BOND => 10,
        UNBOND => 2,
        SENSE | ENRG | TEMP | SHARE => 1,
        ALLOC | DIVIDE => 30,
        DIGEST => 2, // per byte
        TUMBLE => 2,
        FREE | POOL | RAND | SLEEP => 1,
        HALT => 0,
        0x70..=0x7F => 1,
        _ => 1, // jumps
    }
}

pub fn mnemonic(b: u8) -> String {
    use op::*;
    let s = match b {
        NOP => "NOP",
        DUP => "DUP",
        DROP => "DROP",
        SWAP => "SWAP",
        OVER => "OVER",
        INC => "INC",
        DEC => "DEC",
        ADD => "ADD",
        SUB => "SUB",
        SHL4 => "SHL4",
        NEG => "NEG",
        AND => "AND",
        SKIPNZ => "SKIPNZ",
        SKIPZ => "SKIPZ",
        JREL => "JREL",
        JNZ => "JNZ",
        RD => "RD",
        WR => "WR",
        CPY => "CPY",
        H0INC => "H0+",
        H0DEC => "H0-",
        H1INC => "H1+",
        H1DEC => "H1-",
        SETH0 => "SETH0",
        SETH1 => "SETH1",
        GETH0 => "GETH0",
        GETH1 => "GETH1",
        XH => "XH",
        LEN => "LEN",
        GROW => "GROW",
        SHRINK => "SHRINK",
        PLEN => "PLEN",
        0x20 => "YAW_L",
        0x21 => "YAW_D",
        0x22 => "PITCH_L",
        0x23 => "PITCH_D",
        0x24 => "ROLL_L",
        0x25 => "ROLL_D",
        0x26 => "BONDS_L",
        0x27 => "BONDS_D",
        0x28 => "SENS_L",
        0x29 => "SENS_D",
        0x2A => "STEP_L",
        0x2B => "STEP_D",
        0x2C => "CAT_L",
        0x2D => "CAT_D",
        0x2E => "TAKE_L",
        0x2F => "TAKE_D",
        0x30 => "EMIT_L",
        0x31 => "EMIT_D",
        0x32 => "SIG_L",
        0x33 => "SIG_D",
        0x34 => "CHEM_L",
        0x35 => "CHEM_D",
        0x36..=0x3F => {
            return format!("FILL_{}{}", if b & 1 == 0 { "L" } else { "D" }, (b - 0x36) / 2);
        }
        0x40..=0x5F => return format!("PUSH {}", b - 0x40),
        MOVE => "MOVE",
        BOND => "BOND",
        UNBOND => "UNBOND",
        SENSE => "SENSE",
        ENRG => "ENRG",
        TEMP => "TEMP",
        SHARE => "SHARE",
        ALLOC => "ALLOC",
        DIVIDE => "DIVIDE",
        DIGEST => "DIGEST",
        TUMBLE => "TUMBLE",
        FREE => "FREE",
        POOL => "POOL",
        RAND => "RAND",
        SLEEP => "SLEEP",
        HALT => "HALT",
        0x70..=0x7F => return format!("FILL{:X}", b & 0xF),
        0x80..=0xBF => return format!("JMP +{}", 2 + (b & 0x3F) as u32),
        _ => return format!("JMP -{}", 1 + (b & 0x3F) as u32),
    };
    s.to_string()
}

/// Opcode group, used by the viewer colouring.
pub fn group(b: u8) -> u8 {
    match b {
        0 => 0,
        0x01..=0x0F => 1,
        0x10..=0x1F => 2,
        0x20..=0x3F => 3,
        0x40..=0x5F => 4,
        0x60..=0x6F => 5,
        0x70..=0x7F => 6,
        _ => 7,
    }
}

pub fn parse_opcode_name(name: &str) -> Option<u8> {
    let up = name.trim().to_ascii_uppercase();
    if let Some(h) = up.strip_prefix("0X") {
        return u8::from_str_radix(h, 16).ok();
    }
    (0..=255u8).find(|&b| mnemonic(b) == up)
}

pub fn parse_hex_genome(s: &str) -> anyhow::Result<Vec<u8>> {
    let clean: String = s.chars().filter(|c| c.is_ascii_hexdigit()).collect();
    anyhow::ensure!(clean.len() % 2 == 0, "odd number of hex digits");
    (0..clean.len() / 2)
        .map(|i| u8::from_str_radix(&clean[2 * i..2 * i + 2], 16).map_err(|e| e.into()))
        .collect()
}

pub fn to_hex(t: &[u8]) -> String {
    t.iter().map(|b| format!("{b:02x}")).collect::<Vec<_>>().join(" ")
}

pub const MINIMAL_REPLICATOR: [u8; 4] = [op::CPY, op::H0INC, op::H1INC, 0xC2];
