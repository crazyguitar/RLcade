#[allow(unused_macros)]
macro_rules! nes_info {
    ($($arg:tt)*) => {
        println!("[INFO] {}", format!($($arg)*));
    }
}

#[allow(unused_macros)]
macro_rules! nes_warn {
    ($($arg:tt)*) => {
        eprintln!("[WARN] {}", format!($($arg)*));
    }
}

#[allow(unused_macros)]
macro_rules! nes_error {
    ($($arg:tt)*) => {
        eprintln!("[ERROR] {}", format!($($arg)*));
    }
}

pub(crate) use nes_error;
pub(crate) use nes_info;
pub(crate) use nes_warn;
