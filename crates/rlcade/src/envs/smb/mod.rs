pub mod base;
pub mod wrappers;

pub use base::{StepResult, SuperMarioBrosEnv};
pub use wrappers::*;

/// Common env trait — all wrappers implement this.
///
/// Requires `Send` so `Box<dyn Env>` can be moved to a worker thread for
/// vectorized envs; every concrete impl holds only `Send` state.
pub trait Env: Send {
    fn step(&mut self, action: usize) -> StepResult;
    fn reset(&mut self) -> StepResult;
    fn screen_rgb(&self) -> Vec<u8>;
    fn screen_buffer(&self) -> &[u32];
    /// Raw screen from the base emulator, bypassing max-pooling. For human rendering.
    #[rustfmt::skip]
    fn raw_screen(&self) -> &[u32] { self.screen_buffer() }
    fn screen_width(&self) -> usize;
    fn screen_height(&self) -> usize;
    fn life(&self) -> u8;
    /// Stacked observation (if applicable).
    fn obs(&self) -> &[f32];
    /// Number of valid actions the env accepts (indices 0..num_actions).
    fn num_actions(&self) -> usize;
}

/// Wrappers that delegate passthrough methods to an inner `Env`.
///
/// Implement `EnvWrapper` instead of `Env` when the wrapper only
/// customizes `step`/`reset` and forwards everything else.
/// The blanket impl below provides the remaining `Env` methods.
pub trait EnvWrapper: Send {
    type Inner: Env;
    fn inner(&self) -> &Self::Inner;
    fn inner_mut(&mut self) -> &mut Self::Inner;

    fn step(&mut self, action: usize) -> StepResult;
    fn reset(&mut self) -> StepResult;
}

#[rustfmt::skip]
impl<T: EnvWrapper> Env for T {
    fn step(&mut self, action: usize) -> StepResult { EnvWrapper::step(self, action) }
    fn reset(&mut self) -> StepResult { EnvWrapper::reset(self) }
    fn screen_rgb(&self) -> Vec<u8> { self.inner().screen_rgb() }
    fn screen_buffer(&self) -> &[u32] { self.inner().screen_buffer() }
    fn raw_screen(&self) -> &[u32] { self.inner().raw_screen() }
    fn screen_width(&self) -> usize { self.inner().screen_width() }
    fn screen_height(&self) -> usize { self.inner().screen_height() }
    fn life(&self) -> u8 { self.inner().life() }
    fn obs(&self) -> &[f32] { self.inner().obs() }
    fn num_actions(&self) -> usize { self.inner().num_actions() }
}
