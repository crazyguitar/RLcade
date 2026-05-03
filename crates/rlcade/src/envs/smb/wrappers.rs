//! Gym-style wrappers — each wraps an inner `Env` and adds one behavior.

use nes_core::core::screen::to_rgb;

use super::{Env, EnvWrapper, StepResult};

// MaxAndSkip
//
// Repeats the same action for `skip` frames, accumulating reward.
// Stores the last two frames as u32 pixel buffers for max-pooling.
// The max-pooled buffer is cached and returned by `screen_buffer()`
// so downstream wrappers (WarpFrame) consume it without any RGB conversion.

pub struct MaxAndSkip<E: Env> {
    inner: E,
    skip: usize,
    /// Packed last-two frames — `[ buf0_pixels... | buf1_pixels... ]`.
    /// One contiguous allocation keeps buf0 and buf1 adjacent in memory,
    /// which helps the prefetcher in `max_pool` (reads both in lock-step).
    buf: Vec<u32>,
    frame_size: usize,
    pooled: Vec<u32>,
    /// Counts frames within the current skip window (0..skip).
    counter: usize,
    /// Accumulated reward across the skip window.
    total_reward: f32,
}

impl<E: Env> MaxAndSkip<E> {
    /// # Panics
    /// Panics if `skip < 2`; max-pooling needs at least two frames.
    pub fn new(inner: E, skip: usize) -> Self {
        assert!(skip >= 2, "MaxAndSkip: skip must be >= 2, got {}", skip);
        let frame_size = inner.screen_width() * inner.screen_height();
        Self {
            inner,
            skip,
            buf: vec![0u32; frame_size * 2],
            frame_size,
            pooled: vec![0u32; frame_size],
            counter: 0,
            total_reward: 0.0,
        }
    }

    /// Compute element-wise max of the two frame buffers into `self.pooled`.
    fn max_pool(&mut self) {
        let (buf0, buf1) = self.buf.split_at(self.frame_size);
        for (dst, (&a, &b)) in self.pooled.iter_mut().zip(buf0.iter().zip(buf1.iter())) {
            *dst = a.max(b);
        }
    }
}

impl<E: Env> Env for MaxAndSkip<E> {
    fn step(&mut self, action: usize) -> StepResult {
        let mut result = self.inner.step(action);

        // Store last 2 frames for max-pooling (eliminates NES sprite flicker)
        let i = self.counter;
        if i >= self.skip - 2 {
            let slot = i - (self.skip - 2);
            let start = slot * self.frame_size;
            let end = start + self.frame_size;
            self.buf[start..end].copy_from_slice(self.inner.screen_buffer());
        }
        self.total_reward += result.reward;
        self.counter += 1;

        if result.terminated || self.counter >= self.skip {
            self.max_pool();
            result.reward = self.total_reward;
            result.ready = true;
            self.counter = 0;
            self.total_reward = 0.0;
        }
        result
    }

    fn reset(&mut self) -> StepResult {
        self.counter = 0;
        self.total_reward = 0.0;
        let result = self.inner.reset();
        // Prime both slots and the pooled cache with the fresh frame so the
        // first `screen_buffer()` read after reset reflects the new episode
        // (otherwise WarpFrame would consume a zero/stale pooled frame).
        let fresh = self.inner.screen_buffer();
        self.buf[..self.frame_size].copy_from_slice(fresh);
        self.buf[self.frame_size..].copy_from_slice(fresh);
        self.pooled.copy_from_slice(fresh);
        result
    }

    fn screen_rgb(&self) -> Vec<u8> {
        let w = self.inner.screen_width();
        let h = self.inner.screen_height();
        to_rgb(&self.pooled, w, h)
    }

    #[rustfmt::skip]
    fn screen_buffer(&self) -> &[u32] { &self.pooled }
    #[rustfmt::skip]
    fn raw_screen(&self) -> &[u32] { self.inner.screen_buffer() }
    #[rustfmt::skip]
    fn screen_width(&self) -> usize { self.inner.screen_width() }
    #[rustfmt::skip]
    fn screen_height(&self) -> usize { self.inner.screen_height() }
    #[rustfmt::skip]
    fn life(&self) -> u8 { self.inner.life() }
    #[rustfmt::skip]
    fn obs(&self) -> &[f32] { self.inner.obs() }
    #[rustfmt::skip]
    fn num_actions(&self) -> usize { self.inner.num_actions() }
}

// EpisodicLife

pub struct EpisodicLife<E: Env> {
    inner: E,
    lives: u8,
    real_done: bool,
}

impl<E: Env> EpisodicLife<E> {
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            lives: 0,
            real_done: true,
        }
    }
}

impl<E: Env> EnvWrapper for EpisodicLife<E> {
    type Inner = E;
    fn inner(&self) -> &E {
        &self.inner
    }
    fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }

    fn step(&mut self, action: usize) -> StepResult {
        let mut result = self.inner.step(action);
        self.real_done = result.terminated;
        let lives = self.inner.life();
        if lives > 0 && lives < self.lives {
            result.terminated = true;
        }
        self.lives = lives;
        result
    }

    fn reset(&mut self) -> StepResult {
        if self.real_done {
            let result = self.inner.reset();
            self.lives = self.inner.life();
            return result;
        }
        // noop step to advance from the lost-life state. The inner env may
        // still be mid-death-animation, so clear `terminated` and `reward`
        // — reset() must return a clean episode-start frame.
        let mut result = self.inner.step(0);
        result.terminated = false;
        result.reward = 0.0;
        self.lives = self.inner.life();
        result
    }
}

// CustomReward
//
// The default SMB reward (x-position delta + time penalty + death) is sparse:
// the agent can wander for hundreds of frames with near-zero feedback. These
// six shaping signals provide denser intermediate rewards so the agent learns
// faster without changing the optimal policy (reach the flag).
//
//   1. Velocity bonus — rewards rightward *speed*, not just displacement.
//      Without this the agent learns to inch forward; with it, running and
//      maintaining momentum is explicitly encouraged.
//
//   2. Novelty bonus — extra reward for reaching x-positions never seen in
//      the current episode. Prevents the agent from farming reward by
//      walking back and forth over the same ground.
//
//   3. Time penalty — small constant cost per step. Breaks ties between
//      policies that reach the same x-position but at different speeds,
//      biasing toward faster completion.
//
//   4. Stagnation penalty — escalating cost when max_x hasn't advanced for
//      many steps. After a grace period (50 steps), the penalty grows
//      linearly up to a cap, discouraging back-and-forth oscillation
//      without stacking too harshly with the death penalty.
//
//   5. Coin/powerup bonus — rewards collecting coins and gaining powerup
//      status (small → big → fire). Teaches the agent that powerups
//      improve survivability, which transfers to harder stages.
//
//   6. Death penalty scaling — dying early (low x) is penalized more than
//      dying late (high x). This avoids the "give up immediately" failure
//      mode where the agent learns that dying quickly minimizes cumulative
//      time penalty.

/// Approximate max x-position reachable in a single SMB stage.
const MAX_X_APPROX: f32 = 3200.0;
/// Max pixels Mario can move per frame-skip; anything larger is a pipe/warp.
const X_DELTA_MAX: u16 = 5;
/// Reward per pixel of rightward movement.
const VELOCITY_COEF: f32 = 0.01;
/// Reward per pixel of new max-x reached.
const NOVELTY_COEF: f32 = 0.02;
/// Constant cost per step to discourage idling.
const TIME_PENALTY: f32 = -0.01;
/// Reward per coin collected.
const COIN_REWARD: f32 = 0.5;
/// Reward for gaining a powerup level (small→big or big→fire).
const POWERUP_REWARD: f32 = 1.0;
/// Reward for reaching the flag.
const FLAG_REWARD: f32 = 5.0;
/// Max death penalty (applied when dying at x=0; scales down with progress).
const DEATH_PENALTY_MAX: f32 = 5.0;
/// Steps without max_x advance before stagnation penalty kicks in.
const STAGNATION_GRACE: u32 = 50;
/// Penalty per step beyond the grace period (grows linearly).
const STAGNATION_COEF: f32 = 0.005;
/// Maximum stagnation penalty per step (caps the linear growth).
const STAGNATION_MAX: f32 = 0.5;

pub struct CustomReward<E: Env> {
    inner: E,
    prev_x: u16,
    max_x: u16,
    prev_coins: u16,
    prev_status: u8,
    first_step: bool,
    /// Steps since last max_x advance (for stagnation penalty).
    stagnation_steps: u32,
}

impl<E: Env> CustomReward<E> {
    pub fn new(inner: E) -> Self {
        Self {
            inner,
            prev_x: 0,
            max_x: 0,
            prev_coins: 0,
            prev_status: 0,
            first_step: true,
            stagnation_steps: 0,
        }
    }

    /// Reward proportional to rightward movement since last step.
    /// Ignores large jumps (pipes/warps) where |dx| exceeds the threshold.
    fn velocity_bonus(&self, x_pos: u16) -> f32 {
        let dx = x_pos as f32 - self.prev_x as f32;
        if dx.abs() > X_DELTA_MAX as f32 {
            0.0
        } else {
            dx * VELOCITY_COEF
        }
    }

    /// Extra reward for reaching a new max x-position in the episode.
    /// Ignores large jumps (pipes/warps) to avoid spurious spikes.
    fn novelty_bonus(&mut self, x_pos: u16) -> f32 {
        if x_pos <= self.max_x {
            return 0.0;
        }
        let delta = x_pos - self.max_x;
        // Discard pipe/warp jumps entirely — don't advance max_x, or future
        // novelty and stagnation tracking would stay poisoned for the rest
        // of the episode.
        if delta > X_DELTA_MAX {
            return 0.0;
        }
        self.max_x = x_pos;
        delta as f32 * NOVELTY_COEF
    }

    /// Update the stagnation counter: reset on forward progress, increment otherwise.
    fn update_stagnation(&mut self, x_pos: u16) {
        if x_pos > self.max_x {
            self.stagnation_steps = 0;
        } else {
            self.stagnation_steps += 1;
        }
    }

    /// Reward for collecting coins or gaining powerup status.
    /// Handles coin counter wrapping (99 → 0) by treating a decrease as +1.
    fn coin_powerup_bonus(&self, coins: u16, status: u8) -> f32 {
        let coin_delta = match coins.cmp(&self.prev_coins) {
            std::cmp::Ordering::Greater => coins - self.prev_coins,
            std::cmp::Ordering::Less => 1, // counter wrapped (e.g. 99 → 0)
            std::cmp::Ordering::Equal => 0,
        };
        let coin_reward = coin_delta as f32 * COIN_REWARD;
        let powerup_reward = if status > self.prev_status {
            POWERUP_REWARD
        } else {
            0.0
        };
        coin_reward + powerup_reward
    }

    /// Terminal reward: flag bonus on success, scaled penalty on death.
    fn terminal_reward(x_pos: u16, flag_get: bool) -> f32 {
        if flag_get {
            FLAG_REWARD
        } else {
            let progress = (x_pos as f32 / MAX_X_APPROX).min(1.0);
            -DEATH_PENALTY_MAX * (1.0 - progress)
        }
    }

    /// Escalating penalty when max_x hasn't advanced for too long, capped
    /// at STAGNATION_MAX so it doesn't stack too harshly with the death penalty.
    fn stagnation_penalty(&self) -> f32 {
        if self.stagnation_steps > STAGNATION_GRACE {
            let overshoot = (self.stagnation_steps - STAGNATION_GRACE) as f32;
            (-STAGNATION_COEF * overshoot).max(-STAGNATION_MAX)
        } else {
            0.0
        }
    }
}

impl<E: Env> EnvWrapper for CustomReward<E> {
    type Inner = E;
    fn inner(&self) -> &E {
        &self.inner
    }
    fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }

    fn step(&mut self, action: usize) -> StepResult {
        let mut result = self.inner.step(action);

        if self.first_step {
            self.prev_x = result.x_pos;
            self.max_x = result.x_pos;
            self.prev_coins = result.coins;
            self.prev_status = result.status;
            // Reset stagnation on the first frame so the agent isn't charged
            // one stagnation step for "no forward progress" when max_x was
            // just initialised from x_pos.
            self.stagnation_steps = 0;
            self.first_step = false;
        }

        // update_stagnation must run BEFORE novelty_bonus, because
        // novelty_bonus mutates max_x — we need to compare x_pos against
        // the previous step's max to detect forward progress.
        self.update_stagnation(result.x_pos);

        let mut reward = self.velocity_bonus(result.x_pos);
        reward += self.novelty_bonus(result.x_pos);
        reward += TIME_PENALTY;
        reward += self.stagnation_penalty();
        reward += self.coin_powerup_bonus(result.coins, result.status);
        if result.terminated {
            reward += Self::terminal_reward(result.x_pos, result.flag_get);
        }

        self.prev_x = result.x_pos;
        self.prev_coins = result.coins;
        self.prev_status = result.status;
        result.reward = reward;
        result
    }

    fn reset(&mut self) -> StepResult {
        self.prev_x = 0;
        self.max_x = 0;
        self.prev_coins = 0;
        self.prev_status = 0;
        self.first_step = true;
        self.stagnation_steps = 0;
        self.inner.reset()
    }
}

// ClipReward

pub struct ClipReward<E: Env> {
    inner: E,
}

impl<E: Env> ClipReward<E> {
    pub fn new(inner: E) -> Self {
        Self { inner }
    }
}

impl<E: Env> EnvWrapper for ClipReward<E> {
    type Inner = E;
    fn inner(&self) -> &E {
        &self.inner
    }
    fn inner_mut(&mut self) -> &mut E {
        &mut self.inner
    }

    fn step(&mut self, action: usize) -> StepResult {
        let mut result = self.inner.step(action);
        result.reward = result.reward.signum();
        result
    }

    fn reset(&mut self) -> StepResult {
        self.inner.reset()
    }
}

// WarpFrame (grayscale + resize 84×84 + scale to [0,1])

const FRAME_W: usize = 84;
const FRAME_H: usize = 84;

pub struct WarpFrame<E: Env> {
    inner: E,
    frame: Vec<f32>,
    x_map: Vec<ResizeAxis>,
    y_map: Vec<ResizeAxis>,
}

impl<E: Env> WarpFrame<E> {
    pub fn new(inner: E) -> Self {
        let x_map = resize_axis_map(inner.screen_width(), FRAME_W);
        let y_map = resize_axis_map(inner.screen_height(), FRAME_H);
        Self {
            inner,
            frame: vec![0.0; FRAME_H * FRAME_W],
            x_map,
            y_map,
        }
    }

    fn warp(&mut self) {
        let pixels = self.inner.screen_buffer();
        gray_resize(
            pixels,
            self.inner.screen_width(),
            &mut self.frame,
            FRAME_W,
            &self.x_map,
            &self.y_map,
        );
    }
}

// ITU-R BT.601 luma coefficients (fixed-point, sum = 256)
const LUMA_R: u32 = 77;
const LUMA_G: u32 = 150;
const LUMA_B: u32 = 29;

/// Convert a packed 0x00RRGGBB pixel to luma in [0, 1].
#[inline]
fn luma(pixel: u32) -> f32 {
    let r = (pixel >> 16) & 0xFF;
    let g = (pixel >> 8) & 0xFF;
    let b = pixel & 0xFF;
    (LUMA_R * r + LUMA_G * g + LUMA_B * b) as f32 * (1.0 / 255.0 / 256.0)
}

#[derive(Clone, Copy)]
struct ResizeAxis {
    i0: usize,
    i1: usize,
    w0: f32,
    w1: f32,
}

fn resize_axis_map(src: usize, dst: usize) -> Vec<ResizeAxis> {
    let ratio = src as f32 / dst as f32;
    (0..dst)
        .map(|di| {
            let si = di as f32 * ratio;
            let i0 = (si as usize).min(src - 1);
            let i1 = (i0 + 1).min(src - 1);
            let w1 = si - i0 as f32;
            ResizeAxis {
                i0,
                i1,
                w0: 1.0 - w1,
                w1,
            }
        })
        .collect()
}

/// Combined grayscale + bilinear resize from `src` (sw×sh packed u32) into
/// `dst` (dw×dh f32 in [0, 1]).
///
/// Source indices and interpolation weights are precomputed by `WarpFrame`,
/// leaving the hot path to do only row selection, luma conversion, and blends.
#[inline]
fn gray_resize(
    src: &[u32],
    sw: usize,
    dst: &mut [f32],
    dw: usize,
    x_map: &[ResizeAxis],
    y_map: &[ResizeAxis],
) {
    for (dy, y) in y_map.iter().enumerate() {
        let row0 = &src[y.i0 * sw..y.i0 * sw + sw];
        let row1 = &src[y.i1 * sw..y.i1 * sw + sw];

        let dst_row = &mut dst[dy * dw..dy * dw + dw];
        for (dst_pixel, x) in dst_row.iter_mut().zip(x_map.iter()) {
            let top = luma(row0[x.i0]) * x.w0 + luma(row0[x.i1]) * x.w1;
            let bot = luma(row1[x.i0]) * x.w0 + luma(row1[x.i1]) * x.w1;
            *dst_pixel = top * y.w0 + bot * y.w1;
        }
    }
}

impl<E: Env> Env for WarpFrame<E> {
    fn step(&mut self, action: usize) -> StepResult {
        let result = self.inner.step(action);
        if result.ready {
            self.warp();
        }
        result
    }

    fn reset(&mut self) -> StepResult {
        let result = self.inner.reset();
        self.warp();
        result
    }

    #[rustfmt::skip]
    fn screen_rgb(&self) -> Vec<u8> { self.inner.screen_rgb() }
    #[rustfmt::skip]
    fn screen_buffer(&self) -> &[u32] { self.inner.screen_buffer() }
    #[rustfmt::skip]
    fn raw_screen(&self) -> &[u32] { self.inner.raw_screen() }
    #[rustfmt::skip]
    fn screen_width(&self) -> usize { self.inner.screen_width() }
    #[rustfmt::skip]
    fn screen_height(&self) -> usize { self.inner.screen_height() }
    #[rustfmt::skip]
    fn life(&self) -> u8 { self.inner.life() }
    #[rustfmt::skip]
    fn obs(&self) -> &[f32] { &self.frame }
    #[rustfmt::skip]
    fn num_actions(&self) -> usize { self.inner.num_actions() }
}

// FrameStack

pub struct FrameStack<E: Env> {
    inner: E,
    k: usize,
    stacked: Vec<f32>,
    frame_size: usize,
}

impl<E: Env> FrameStack<E> {
    /// # Panics
    /// Panics if `k == 0`; at least one frame must be stacked.
    pub fn new(inner: E, k: usize) -> Self {
        assert!(k > 0, "FrameStack: k must be >= 1, got 0");
        let frame_size = FRAME_H * FRAME_W;
        Self {
            inner,
            k,
            stacked: vec![0.0; k * frame_size],
            frame_size,
        }
    }

    fn push(&mut self) {
        let obs = self.inner.obs();
        if self.k > 1 {
            self.stacked.copy_within(self.frame_size.., 0);
        }
        let dst = (self.k - 1) * self.frame_size;
        self.stacked[dst..dst + self.frame_size].copy_from_slice(obs);
    }

    fn fill(&mut self) {
        let obs = self.inner.obs();
        for i in 0..self.k {
            let dst = i * self.frame_size;
            self.stacked[dst..dst + self.frame_size].copy_from_slice(obs);
        }
    }
}

impl<E: Env> Env for FrameStack<E> {
    fn step(&mut self, action: usize) -> StepResult {
        let result = self.inner.step(action);
        if result.ready {
            self.push();
        }
        result
    }

    fn reset(&mut self) -> StepResult {
        let result = self.inner.reset();
        self.fill();
        result
    }

    #[rustfmt::skip]
    fn screen_rgb(&self) -> Vec<u8> { self.inner.screen_rgb() }
    #[rustfmt::skip]
    fn screen_buffer(&self) -> &[u32] { self.inner.screen_buffer() }
    #[rustfmt::skip]
    fn raw_screen(&self) -> &[u32] { self.inner.raw_screen() }
    #[rustfmt::skip]
    fn screen_width(&self) -> usize { self.inner.screen_width() }
    #[rustfmt::skip]
    fn screen_height(&self) -> usize { self.inner.screen_height() }
    #[rustfmt::skip]
    fn life(&self) -> u8 { self.inner.life() }
    #[rustfmt::skip]
    fn obs(&self) -> &[f32] { &self.stacked }
    #[rustfmt::skip]
    fn num_actions(&self) -> usize { self.inner.num_actions() }
}
