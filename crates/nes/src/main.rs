extern crate nes_core;

use nes_core::core::nes::Nes;
use std::env;

fn main() {
    let args: Vec<String> = env::args().collect();
    let rom_path = args
        .get(1)
        .map_or("games/super-mario-bros.nes", |s| s.as_str());

    let mut nes = Nes::new(rom_path);
    nes.run();
}
