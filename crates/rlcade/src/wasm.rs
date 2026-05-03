//! WebAssembly bindings for the NES emulator and SMB environment.

use wasm_bindgen::prelude::*;

use nes_core::core::emulator::Emulator;
use nes_core::core::screen::to_rgb;

use crate::envs::smb::base::SuperMarioBrosEnv;
use crate::envs::smb::wrappers::{
    ClipReward, CustomReward, EpisodicLife, FrameStack, MaxAndSkip, WarpFrame,
};
use crate::envs::smb::{Env, StepResult};

// WasmNes: raw emulator

/// Raw NES emulator exposed to JavaScript.
#[wasm_bindgen]
pub struct WasmNes {
    emu: Emulator,
}

#[wasm_bindgen]
impl WasmNes {
    /// Create a new NES emulator from ROM bytes.
    #[wasm_bindgen(constructor)]
    pub fn new(rom: &[u8]) -> Result<WasmNes, JsValue> {
        let mut emu = Emulator::new();
        emu.load_bytes(rom)
            .map_err(|e| JsValue::from_str(&e.to_string()))?;
        emu.reset();
        Ok(WasmNes { emu })
    }

    /// Reset the emulator.
    pub fn reset(&mut self) {
        self.emu.reset();
    }

    /// Advance one frame with the given joypad states.
    pub fn step_frame(&mut self, p1: u8, p2: u8) {
        self.emu.set_joypad(0, p1);
        self.emu.set_joypad(1, p2);
        self.emu.step_frame();
    }

    /// Get the screen as RGB bytes (256 x 240 x 3).
    pub fn screen_rgb(&self) -> Vec<u8> {
        to_rgb(
            self.emu.screen_buffer(),
            self.emu.width(),
            self.emu.height(),
        )
    }

    pub fn screen_width(&self) -> usize {
        self.emu.width()
    }

    pub fn screen_height(&self) -> usize {
        self.emu.height()
    }
}

// WasmSmbEnv: full SMB environment with wrappers

/// SMB environment step result exposed to JavaScript.
#[wasm_bindgen]
pub struct WasmStepResult {
    pub reward: f32,
    pub terminated: bool,
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

impl From<StepResult> for WasmStepResult {
    fn from(r: StepResult) -> Self {
        Self {
            reward: r.reward,
            terminated: r.terminated,
            ready: r.ready,
            coins: r.coins,
            flag_get: r.flag_get,
            life: r.life,
            score: r.score,
            stage: r.stage,
            status: r.status,
            time: r.time,
            world: r.world,
            x_pos: r.x_pos,
            y_pos: r.y_pos,
        }
    }
}

/// Full Super Mario Bros environment with wrapper chain, for WASM.
#[wasm_bindgen]
pub struct WasmSmbEnv {
    env: Box<dyn Env>,
}

fn build_env_from_bytes(
    rom: &[u8],
    actions: Vec<u8>,
    world: Option<u8>,
    stage: Option<u8>,
    skip: usize,
    episodic_life: bool,
    custom_reward: bool,
    clip_rewards: bool,
    frame_stack: usize,
) -> std::io::Result<Box<dyn Env>> {
    let base = SuperMarioBrosEnv::new_from_bytes(rom, actions, world, stage)?;
    let skipped = MaxAndSkip::new(base, skip.max(2));
    let fs = frame_stack.max(1);

    let env: Box<dyn Env> = if episodic_life {
        build_obs_chain(EpisodicLife::new(skipped), custom_reward, clip_rewards, fs)
    } else {
        build_obs_chain(skipped, custom_reward, clip_rewards, fs)
    };
    Ok(env)
}

fn build_obs_chain(
    inner: impl Env + 'static,
    custom_reward: bool,
    clip_rewards: bool,
    frame_stack: usize,
) -> Box<dyn Env> {
    if custom_reward {
        let env = CustomReward::new(inner);
        let env = WarpFrame::new(env);
        Box::new(FrameStack::new(env, frame_stack))
    } else if clip_rewards {
        let env = ClipReward::new(inner);
        let env = WarpFrame::new(env);
        Box::new(FrameStack::new(env, frame_stack))
    } else {
        let env = WarpFrame::new(inner);
        Box::new(FrameStack::new(env, frame_stack))
    }
}

#[wasm_bindgen]
impl WasmSmbEnv {
    /// Create a new SMB environment from ROM bytes.
    ///
    /// `actions` is a JS array of joypad bitmasks (e.g. [0, 0x80, 0x81, ...]).
    #[wasm_bindgen(constructor)]
    pub fn new(
        rom: &[u8],
        actions: Vec<u8>,
        world: u8,
        stage: u8,
        skip: usize,
        episodic_life: bool,
        custom_reward: bool,
        clip_rewards: bool,
        frame_stack: usize,
    ) -> Result<WasmSmbEnv, JsError> {
        let env = build_env_from_bytes(
            rom,
            actions,
            Some(world),
            Some(stage),
            skip,
            episodic_life,
            custom_reward,
            clip_rewards,
            frame_stack,
        )
        .map_err(|e| JsError::new(&e.to_string()))?;
        Ok(WasmSmbEnv { env })
    }

    /// Reset the environment. Returns the step result info.
    pub fn reset(&mut self) -> WasmStepResult {
        self.env.reset().into()
    }

    /// Step the environment with the given action index.
    pub fn step(&mut self, action: usize) -> WasmStepResult {
        self.env.step(action).into()
    }

    /// Get the current observation as a flat f32 array (frame_stack * 84 * 84).
    pub fn obs(&self) -> Vec<f32> {
        self.env.obs().to_vec()
    }

    /// Get the raw screen as RGB bytes (256 x 240 x 3).
    pub fn screen_rgb(&self) -> Vec<u8> {
        self.env.screen_rgb()
    }

    pub fn screen_width(&self) -> usize {
        self.env.screen_width()
    }

    pub fn screen_height(&self) -> usize {
        self.env.screen_height()
    }

    pub fn life(&self) -> u8 {
        self.env.life()
    }
}
