//! Base Super Mario Bros environment — NES emulator + game logic.

use nes_core::core::emulator::Emulator;
use nes_core::core::screen::to_rgb;

use super::Env;

// RAM addresses

const PLAYER_STATE: usize = 0x000E;
const PLAYER_FLOAT_STATE: usize = 0x001D;
const ENEMY_TYPE: [usize; 5] = [0x0016, 0x0017, 0x0018, 0x0019, 0x001A];
const Y_VIEWPORT: usize = 0x00B5;
const X_SCROLL_PAGE: usize = 0x6D;
const X_POSITION_ON_PAGE: usize = 0x86;
const Y_PIXEL: usize = 0x03B8;
const LIVES: usize = 0x075A;
const STAGE: usize = 0x075C;
const WORLD: usize = 0x075F;
const AREA: usize = 0x0760;
const GAMEPLAY_MODE: usize = 0x0770;
const PLAYER_STATUS: usize = 0x0756;
const CHANGE_AREA_TIMER: usize = 0x06DE;
const PRELEVEL_TIMER: usize = 0x07A0;
const SCORE: usize = 0x07DE;
const COINS: usize = 0x07ED;
const TIME: usize = 0x07F8;

const START: u8 = 0x08;
const NOOP: u8 = 0;
const STAGE_OVER_ENEMIES: [u8; 2] = [0x2D, 0x31];
const BUSY_STATES: [u8; 7] = [0x00, 0x01, 0x02, 0x03, 0x04, 0x05, 0x07];
const AREA_OFFSET_WORLDS: [u8; 4] = [1, 2, 4, 7];

const STATE_DYING: u8 = 0x0B;
const STATE_DEAD: u8 = 0x06;
const FLOAT_STATE_FLAGPOLE: u8 = 3;
const GAME_OVER_LIVES: u8 = 0xFF;
const GAMEPLAY_END_OF_WORLD: u8 = 2;

const REWARD_MIN: f32 = -15.0;
const REWARD_MAX: f32 = 15.0;
const DEATH_PENALTY: i32 = -25;
const X_DELTA_THRESHOLD: i32 = 5;
const PAGE_SIZE: u16 = 0x100;
const SCREEN_Y_MAX: u16 = 255;
const TIME_DIGITS: usize = 3;
const COINS_DIGITS: usize = 2;
const SCORE_DIGITS: usize = 6;

/// Step result returned through the wrapper chain.
#[derive(Clone, Default)]
pub struct StepResult {
    pub reward: f32,
    pub terminated: bool,
    /// True when a new observation is ready for the agent (every `skip` frames).
    pub ready: bool,
    pub coins: u16,
    pub flag_get: bool,
    pub life: u8,
    pub score: u32,
    pub stage: u8,
    pub status: u8,
    pub time: u16,
    pub world: u8,
    pub x_pos: u16,
    pub y_pos: u16,
}

// RAM readers

fn read_digits(ram: &[u8], addr: usize, len: usize) -> u32 {
    let mut v: u32 = 0;
    for i in 0..len {
        v = v * 10 + ram[addr + i] as u32;
    }
    v
}

fn time(ram: &[u8]) -> u16 {
    read_digits(ram, TIME, TIME_DIGITS) as u16
}
fn x_position(ram: &[u8]) -> u16 {
    ram[X_SCROLL_PAGE] as u16 * PAGE_SIZE + ram[X_POSITION_ON_PAGE] as u16
}

fn y_position(ram: &[u8]) -> u16 {
    if ram[Y_VIEWPORT] < 1 {
        SCREEN_Y_MAX + (SCREEN_Y_MAX - ram[Y_PIXEL] as u16)
    } else {
        SCREEN_Y_MAX - ram[Y_PIXEL] as u16
    }
}

fn is_dying(ram: &[u8]) -> bool {
    ram[PLAYER_STATE] == STATE_DYING || ram[Y_VIEWPORT] > 1
}
fn is_dead(ram: &[u8]) -> bool {
    ram[PLAYER_STATE] == STATE_DEAD
}
fn is_game_over(ram: &[u8]) -> bool {
    ram[LIVES] == GAME_OVER_LIVES
}
fn is_busy(ram: &[u8]) -> bool {
    BUSY_STATES.contains(&ram[PLAYER_STATE])
}
fn is_world_over(ram: &[u8]) -> bool {
    ram[GAMEPLAY_MODE] == GAMEPLAY_END_OF_WORLD
}

fn is_stage_over(ram: &[u8]) -> bool {
    for &addr in &ENEMY_TYPE {
        if STAGE_OVER_ENEMIES.contains(&ram[addr]) {
            return ram[PLAYER_FLOAT_STATE] == FLOAT_STATE_FLAGPOLE;
        }
    }
    false
}

fn flag_get(ram: &[u8]) -> bool {
    is_world_over(ram) || is_stage_over(ram)
}

fn frame_advance(emu: &mut Emulator, action: u8) {
    emu.set_joypad(0, action);
    emu.step_frame();
}

pub struct SuperMarioBrosEnv {
    emu: Emulator,
    actions: Vec<u8>,
    target_world: Option<u8>,
    target_stage: Option<u8>,
    target_area: Option<u8>,
    time_last: u16,
    x_position_last: u16,
}

impl SuperMarioBrosEnv {
    pub fn new(
        rom: &str,
        actions: Vec<u8>,
        world: Option<u8>,
        stage: Option<u8>,
    ) -> std::io::Result<Self> {
        let mut emu = Emulator::new();
        emu.load(rom)?;
        Ok(Self::init(emu, actions, world, stage))
    }

    pub fn new_from_bytes(
        rom: &[u8],
        actions: Vec<u8>,
        world: Option<u8>,
        stage: Option<u8>,
    ) -> std::io::Result<Self> {
        let mut emu = Emulator::new();
        emu.load_bytes(rom)?;
        Ok(Self::init(emu, actions, world, stage))
    }

    fn init(mut emu: Emulator, actions: Vec<u8>, world: Option<u8>, stage: Option<u8>) -> Self {
        assert!(!actions.is_empty(), "actions must not be empty");
        emu.mute();
        emu.reset();

        let target_area = world.zip(stage).map(|(w, s)| {
            if AREA_OFFSET_WORLDS.contains(&w) && s >= 2 {
                s + 1
            } else {
                s
            }
        });

        Self {
            emu,
            actions,
            target_world: world,
            target_stage: stage,
            target_area,
            time_last: 0,
            x_position_last: 0,
        }
    }

    fn is_single_stage(&self) -> bool {
        self.target_world.is_some()
    }

    fn compute_reward(&mut self) -> f32 {
        let ram = self.emu.ram();
        let x = x_position(ram);
        let x_delta = x as i32 - self.x_position_last as i32;
        self.x_position_last = x;
        let x_reward = if (-X_DELTA_THRESHOLD..=X_DELTA_THRESHOLD).contains(&x_delta) {
            x_delta
        } else {
            0
        };

        let t = time(ram);
        let t_delta = t as i32 - self.time_last as i32;
        self.time_last = t;
        let time_penalty = if t_delta <= 0 { t_delta } else { 0 };

        let death = if is_dying(ram) || is_dead(ram) {
            DEATH_PENALTY
        } else {
            0
        };
        (x_reward + time_penalty + death) as f32
    }

    fn is_episode_done(&self) -> bool {
        let ram = self.emu.ram();
        if self.is_single_stage() {
            is_dying(ram) || is_dead(ram) || flag_get(ram)
        } else {
            is_game_over(ram)
        }
    }

    fn build_step_result(&self) -> StepResult {
        let ram = self.emu.ram();
        StepResult {
            reward: 0.0,
            terminated: false,
            ready: false,
            coins: read_digits(ram, COINS, COINS_DIGITS) as u16,
            flag_get: flag_get(ram),
            life: ram[LIVES],
            score: read_digits(ram, SCORE, SCORE_DIGITS),
            stage: ram[STAGE] + 1,
            status: ram[PLAYER_STATUS],
            time: time(ram),
            world: ram[WORLD] + 1,
            x_pos: x_position(ram),
            y_pos: y_position(ram),
        }
    }

    // RAM hacks

    fn write_stage(&mut self) {
        if let (Some(w), Some(s), Some(a)) =
            (self.target_world, self.target_stage, self.target_area)
        {
            self.emu.ram_mut()[WORLD] = w - 1;
            self.emu.ram_mut()[STAGE] = s - 1;
            self.emu.ram_mut()[AREA] = a - 1;
        }
    }

    fn kill_mario(&mut self) {
        self.emu.ram_mut()[PLAYER_STATE] = STATE_DEAD;
        frame_advance(&mut self.emu, NOOP);
    }

    fn skip_change_area(&mut self) {
        let t = self.emu.ram()[CHANGE_AREA_TIMER];
        if t > 1 && t < 255 {
            self.emu.ram_mut()[CHANGE_AREA_TIMER] = 1;
        }
    }

    fn skip_end_of_world(&mut self) {
        if is_world_over(self.emu.ram()) {
            let t = time(self.emu.ram());
            while time(self.emu.ram()) == t {
                frame_advance(&mut self.emu, NOOP);
            }
        }
    }

    fn skip_occupied_states(&mut self) {
        while is_busy(self.emu.ram()) || is_world_over(self.emu.ram()) {
            self.emu.ram_mut()[PRELEVEL_TIMER] = 0;
            frame_advance(&mut self.emu, NOOP);
        }
    }

    fn did_step(&mut self) {
        if is_dying(self.emu.ram()) {
            self.kill_mario();
        }
        if !self.is_single_stage() {
            self.skip_end_of_world();
        }
        self.skip_change_area();
        self.skip_occupied_states();
    }

    fn skip_start_screen(&mut self) {
        if self.is_single_stage() {
            self.write_stage();
        }
        frame_advance(&mut self.emu, START);
        frame_advance(&mut self.emu, NOOP);

        // Press Start until the game clock begins counting
        while time(self.emu.ram()) == 0 {
            if self.is_single_stage() {
                self.write_stage();
            }
            frame_advance(&mut self.emu, START);
            frame_advance(&mut self.emu, NOOP);
            self.emu.ram_mut()[PRELEVEL_TIMER] = 0;
        }

        // Advance through the pre-level timer countdown
        self.time_last = time(self.emu.ram());
        while time(self.emu.ram()) >= self.time_last {
            self.time_last = time(self.emu.ram());
            frame_advance(&mut self.emu, START);
            frame_advance(&mut self.emu, NOOP);
        }
    }
}

impl Env for SuperMarioBrosEnv {
    fn step(&mut self, action: usize) -> StepResult {
        frame_advance(&mut self.emu, self.actions[action]);
        let mut reward = self.compute_reward();
        reward = reward.clamp(REWARD_MIN, REWARD_MAX);
        let terminated = self.is_episode_done();
        if !terminated {
            self.did_step();
        }
        let mut info = self.build_step_result();
        info.reward = reward;
        info.terminated = terminated;
        info
    }

    fn reset(&mut self) -> StepResult {
        self.emu.reset();
        // Clear game RAM — the NES doesn't clear RAM on reset, so stale
        // values from the previous episode can interfere with write_stage.
        self.emu.ram_mut()[..0x800].fill(0);
        self.time_last = 0;
        self.x_position_last = 0;
        self.skip_start_screen();
        let ram = self.emu.ram();
        self.time_last = time(ram);
        self.x_position_last = x_position(ram);
        self.build_step_result()
    }

    fn screen_rgb(&self) -> Vec<u8> {
        to_rgb(
            self.emu.screen_buffer(),
            self.emu.width(),
            self.emu.height(),
        )
    }

    fn screen_buffer(&self) -> &[u32] {
        self.emu.screen_buffer()
    }
    fn screen_width(&self) -> usize {
        self.emu.width()
    }
    fn screen_height(&self) -> usize {
        self.emu.height()
    }
    fn life(&self) -> u8 {
        self.emu.ram()[LIVES]
    }
    fn obs(&self) -> &[f32] {
        &[]
    }
    fn num_actions(&self) -> usize {
        self.actions.len()
    }
}
