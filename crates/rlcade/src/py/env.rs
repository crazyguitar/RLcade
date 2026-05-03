//! Python bindings for SMB gym environments (single + vectorized).

use std::collections::HashMap;

use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::envs::smb::{Env, StepResult};
use crate::py::Display;
use crate::py::nes::PyScreen;

// Wrapper chain builder

/// Configuration for building an SMB environment wrapper chain.
struct EnvConfig {
    rom: String,
    actions: Vec<u8>,
    world: Option<u8>,
    stage: Option<u8>,
    skip: usize,
    episodic_life: bool,
    custom_reward: bool,
    clip_rewards: bool,
    frame_stack: usize,
}

/// Build the observation pipeline: [reward wrapper] → WarpFrame → FrameStack.
fn build_obs_chain(
    inner: impl Env + 'static,
    custom_reward: bool,
    clip_rewards: bool,
    frame_stack: usize,
) -> Box<dyn Env> {
    use crate::envs::smb::wrappers::{ClipReward, CustomReward, FrameStack, WarpFrame};
    let fs = frame_stack.max(1);
    if custom_reward {
        let env = CustomReward::new(inner);
        let env = WarpFrame::new(env);
        Box::new(FrameStack::new(env, fs))
    } else if clip_rewards {
        let env = ClipReward::new(inner);
        let env = WarpFrame::new(env);
        Box::new(FrameStack::new(env, fs))
    } else {
        let env = WarpFrame::new(inner);
        Box::new(FrameStack::new(env, fs))
    }
}

/// Build the full env wrapper chain from config.
fn build_env(config: EnvConfig) -> PyResult<Box<dyn Env>> {
    use crate::envs::smb::{
        base::SuperMarioBrosEnv,
        wrappers::{EpisodicLife, MaxAndSkip},
    };

    let base = SuperMarioBrosEnv::new(&config.rom, config.actions, config.world, config.stage)
        .map_err(|e| pyo3::exceptions::PyIOError::new_err(e.to_string()))?;
    let skipped = MaxAndSkip::new(base, config.skip.max(2));

    let chained: Box<dyn Env> = if config.episodic_life {
        build_obs_chain(
            EpisodicLife::new(skipped),
            config.custom_reward,
            config.clip_rewards,
            config.frame_stack,
        )
    } else {
        build_obs_chain(
            skipped,
            config.custom_reward,
            config.clip_rewards,
            config.frame_stack,
        )
    };
    Ok(chained)
}

fn step_result_to_dict<'py>(py: Python<'py>, r: &StepResult) -> PyResult<Bound<'py, PyDict>> {
    let d = PyDict::new(py);
    d.set_item("reward", r.reward)?;
    d.set_item("terminated", r.terminated)?;
    d.set_item("ready", r.ready)?;
    d.set_item("coins", r.coins)?;
    d.set_item("flag_get", r.flag_get)?;
    d.set_item("life", r.life)?;
    d.set_item("score", r.score)?;
    d.set_item("stage", r.stage)?;
    d.set_item("status", r.status)?;
    d.set_item("time", r.time)?;
    d.set_item("world", r.world)?;
    d.set_item("x_pos", r.x_pos)?;
    d.set_item("y_pos", r.y_pos)?;
    Ok(d)
}

// NesSmbEnv: single environment

/// Single Super Mario Bros environment.
///
/// Wraps a NES emulator with the standard Atari-style wrapper chain
/// (MaxAndSkip → EpisodicLife → optional reward shaping → WarpFrame →
/// FrameStack) and exposes a Gymnasium-compatible step/reset API.
///
/// Observations are `frame_stack × 84 × 84` float32 tensors in `[0, 1]`.
#[pyclass(name = "NesSmbEnv", unsendable)]
pub struct PyNesSmbEnv {
    env: Box<dyn Env>,
    display: Option<Display>,
}

#[pymethods]
impl PyNesSmbEnv {
    /// Construct a single SMB environment.
    ///
    /// Args:
    ///     rom: path to an iNES ROM file.
    ///     actions: joypad bitmask per discrete action index
    ///         (see `rlcade.envs.smb.ACTIONS`).
    ///     world, stage: target world/stage (1-indexed). `None` for the
    ///         default game flow.
    ///     skip: frames per agent action (max-pooled over the last 2).
    ///         Must be `>= 2`.
    ///     episodic_life: treat loss-of-life as episode end (keeps the
    ///         underlying emulator alive until real game over).
    ///     custom_reward: apply the built-in SMB reward-shaping wrapper.
    ///         Mutually exclusive with `clip_rewards`.
    ///     clip_rewards: clip reward to {-1, 0, +1}. Ignored when
    ///         `custom_reward=True`.
    ///     frame_stack: number of past frames stacked into each obs.
    ///
    /// Raises:
    ///     IOError: ROM file can't be read or isn't a valid iNES cartridge.
    #[new]
    #[pyo3(signature = (rom, actions, world=None, stage=None, skip=4, episodic_life=true, custom_reward=false, clip_rewards=true, frame_stack=4))]
    #[allow(clippy::too_many_arguments)]
    fn new(
        rom: &str,
        actions: Vec<u8>,
        world: Option<u8>,
        stage: Option<u8>,
        skip: usize,
        episodic_life: bool,
        custom_reward: bool,
        clip_rewards: bool,
        frame_stack: usize,
    ) -> PyResult<Self> {
        let config = EnvConfig {
            rom: rom.to_string(),
            actions,
            world,
            stage,
            skip,
            episodic_life,
            custom_reward,
            clip_rewards,
            frame_stack,
        };
        Ok(Self {
            env: build_env(config)?,
            display: None,
        })
    }

    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<(PyObject, Bound<'py, PyDict>)> {
        let info = self.env.reset();
        let arr = numpy::PyArray1::from_slice(py, self.env.obs());
        Ok((arr.into_any().unbind(), step_result_to_dict(py, &info)?))
    }

    fn step_frame<'py>(
        &mut self,
        py: Python<'py>,
        action: usize,
    ) -> PyResult<(PyObject, f32, bool, bool, Bound<'py, PyDict>)> {
        let n = self.env.num_actions();
        if action >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "action {} out of range (expected 0..{})",
                action, n
            )));
        }
        let info = self.env.step(action);
        let arr = numpy::PyArray1::from_slice(py, self.env.obs());
        Ok((
            arr.into_any().unbind(),
            info.reward,
            info.terminated,
            false,
            step_result_to_dict(py, &info)?,
        ))
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        action: usize,
    ) -> PyResult<(PyObject, f32, bool, bool, Bound<'py, PyDict>)> {
        let n = self.env.num_actions();
        if action >= n {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "action {} out of range (expected 0..{})",
                action, n
            )));
        }

        let mut info;
        loop {
            info = self.env.step(action);
            if info.ready {
                break;
            }
        }

        let arr = numpy::PyArray1::from_slice(py, self.env.obs());
        Ok((
            arr.into_any().unbind(),
            info.reward,
            info.terminated,
            false,
            step_result_to_dict(py, &info)?,
        ))
    }

    fn screen(&self) -> PyScreen {
        PyScreen {
            rgb: self.env.screen_rgb(),
        }
    }

    fn render(&mut self) {
        let d = super::ensure_display(
            &mut self.display,
            self.env.screen_width(),
            self.env.screen_height(),
        );
        d.present(self.env.raw_screen());
    }

    fn poll_quit(&mut self) -> bool {
        let d = super::ensure_display(
            &mut self.display,
            self.env.screen_width(),
            self.env.screen_height(),
        );
        d.poll_quit()
    }

    #[getter]
    fn screen_height(&self) -> usize {
        self.env.screen_height()
    }
    #[getter]
    fn screen_width(&self) -> usize {
        self.env.screen_width()
    }
    #[getter]
    fn life(&self) -> u8 {
        self.env.life()
    }
}

// NesVecSmbEnv: vectorized environment with worker pool

/// Run `f` either with the GIL released (when `release` is true) or directly
/// under the current GIL hold. Releasing the GIL costs a handful of atomic
/// ops on every call, so for wait paths where no other Python thread can
/// benefit (e.g. single-worker slice), staying under the GIL is faster.
#[inline]
fn maybe_allow_threads<F, T>(py: Python<'_>, release: bool, f: F) -> T
where
    F: pyo3::marker::Ungil + Send + FnOnce() -> T,
    T: pyo3::marker::Ungil + Send,
{
    if release { py.allow_threads(f) } else { f() }
}

fn worker_gone_error() -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("env worker thread terminated unexpectedly")
}

fn stray_result_error() -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err("received result for worker not in this pool")
}

fn worker_panicked_error(env_id: usize, msg: &str) -> pyo3::PyErr {
    pyo3::exceptions::PyRuntimeError::new_err(format!("env worker {env_id} panicked: {msg}"))
}

fn panic_message(payload: &(dyn std::any::Any + Send)) -> String {
    if let Some(s) = payload.downcast_ref::<&'static str>() {
        (*s).to_string()
    } else if let Some(s) = payload.downcast_ref::<String>() {
        s.clone()
    } else {
        "unknown panic payload".to_string()
    }
}

struct EnvStepOutput {
    /// Which worker produced this output. The pool uses this to reorder
    /// results that arrive in "first-done" order into per-env slot order.
    env_id: usize,
    result: StepResult,
    obs: Vec<f32>,
    final_obs: Option<Vec<f32>>,
}

struct WorkerPanic {
    env_id: usize,
    msg: String,
}

type WorkerOutput = Result<EnvStepOutput, WorkerPanic>;

enum WorkerCmd {
    Step(usize),
    Reset,
}

/// Fill `obs_buf` with the current env observation.
/// Reuses `obs_buf`'s existing capacity — no heap allocation in steady state.
#[inline]
fn fill_obs(env: &dyn Env, obs_buf: &mut Vec<f32>) {
    obs_buf.clear();
    obs_buf.extend_from_slice(env.obs());
}

fn env_step(
    env_id: usize,
    env: &mut dyn Env,
    action: usize,
    mut obs_buf: Vec<f32>,
) -> EnvStepOutput {
    // Loop until the skip window completes or episode terminates.
    let mut result;
    loop {
        result = env.step(action);
        if result.ready {
            break;
        }
    }
    if result.terminated {
        // Capture terminal obs before reset. final_obs is rare (only on episode
        // end), so a fresh allocation here is acceptable — it can't be pooled
        // in lockstep with the main obs ping-pong.
        let final_obs = env.obs().to_vec();
        env.reset();
        fill_obs(env, &mut obs_buf);
        EnvStepOutput {
            env_id,
            result,
            obs: obs_buf,
            final_obs: Some(final_obs),
        }
    } else {
        fill_obs(env, &mut obs_buf);
        EnvStepOutput {
            env_id,
            result,
            obs: obs_buf,
            final_obs: None,
        }
    }
}

fn env_reset(env_id: usize, env: &mut dyn Env, mut obs_buf: Vec<f32>) -> EnvStepOutput {
    let result = env.reset();
    fill_obs(env, &mut obs_buf);
    EnvStepOutput {
        env_id,
        result: StepResult {
            reward: 0.0,
            terminated: false,
            ..result
        },
        obs: obs_buf,
        final_obs: None,
    }
}

fn run_cmd(env_id: usize, env: &mut dyn Env, cmd: WorkerCmd, obs_buf: Vec<f32>) -> WorkerOutput {
    use std::panic::{AssertUnwindSafe, catch_unwind};
    catch_unwind(AssertUnwindSafe(|| match cmd {
        WorkerCmd::Step(action) => env_step(env_id, env, action, obs_buf),
        WorkerCmd::Reset => env_reset(env_id, env, obs_buf),
    }))
    .map_err(|payload| WorkerPanic {
        env_id,
        msg: panic_message(&*payload),
    })
}

fn env_worker_loop(
    env_id: usize,
    mut env: Box<dyn Env>,
    cmd_rx: std::sync::mpsc::Receiver<WorkerCmd>,
    result_tx: std::sync::mpsc::Sender<WorkerOutput>,
    return_rx: std::sync::mpsc::Receiver<Vec<f32>>,
) {
    // Pool of `obs` scratch buffers. Filled by returns from the main thread
    // (via `return_rx`) after it has consumed an output. We start empty; the
    // first few steps allocate, and steady-state uses the pool.
    let mut buf_pool: Vec<Vec<f32>> = Vec::new();
    // Once a command panics, `env` may be in an inconsistent state and we
    // never call back into it. All subsequent commands receive the same
    // panic message so `collect_results` never hangs on a dropped sender.
    let mut poisoned: Option<String> = None;

    while let Ok(cmd) = cmd_rx.recv() {
        while let Ok(buf) = return_rx.try_recv() {
            buf_pool.push(buf);
        }

        let output = match poisoned.as_ref() {
            Some(msg) => Err(WorkerPanic {
                env_id,
                msg: msg.clone(),
            }),
            None => run_cmd(env_id, &mut *env, cmd, buf_pool.pop().unwrap_or_default()),
        };
        if let Err(p) = &output {
            poisoned.get_or_insert_with(|| p.msg.clone());
        }
        // If result_tx is disconnected, Python dropped the env — exit silently.
        if result_tx.send(output).is_err() {
            return;
        }
    }
}

/// A single env worker — owns its command and buffer-return channel ends
/// and the join handle for its thread. Outputs go to a **shared** result
/// channel owned by `WorkerPool`, tagged with `env_id` so the pool can
/// place each result into its correct slot regardless of which worker
/// finished first. Dropping a `Worker` closes `cmd_tx` (worker loop
/// exits) and then joins the thread so destructors run to completion
/// before the parent returns.
struct Worker {
    env_id: usize,
    cmd_tx: std::sync::mpsc::Sender<WorkerCmd>,
    /// Main thread sends consumed `obs` Vecs here; worker reuses them as
    /// scratch to avoid a heap allocation per step.
    return_tx: std::sync::mpsc::Sender<Vec<f32>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

// `mpsc::Sender` is `!Sync` to prevent concurrent `send()` from multiple
// threads. The parent `PyNesVecSmbEnv` is `#[pyclass(unsendable)]`, so all
// access comes from the thread that created it; `send()` is never called
// concurrently. Marking `Worker: Sync` lets us release the GIL via
// `py.allow_threads(…)` across the step/reset blocking wait.
unsafe impl Sync for Worker {}

impl Worker {
    fn spawn(
        env_id: usize,
        env: Box<dyn Env>,
        result_tx: std::sync::mpsc::Sender<WorkerOutput>,
    ) -> Self {
        let (cmd_tx, cmd_rx) = std::sync::mpsc::channel();
        let (return_tx, return_rx) = std::sync::mpsc::channel();
        let handle =
            std::thread::spawn(move || env_worker_loop(env_id, env, cmd_rx, result_tx, return_rx));
        Self {
            env_id,
            cmd_tx,
            return_tx,
            handle: Some(handle),
        }
    }

    fn step(&self, action: usize) {
        let _ = self.cmd_tx.send(WorkerCmd::Step(action));
    }

    fn reset(&self) {
        let _ = self.cmd_tx.send(WorkerCmd::Reset);
    }

    /// Return a consumed obs buffer so the worker can reuse its capacity.
    fn return_buf(&self, buf: Vec<f32>) {
        let _ = self.return_tx.send(buf);
    }
}

impl Drop for Worker {
    fn drop(&mut self) {
        // Field drop order would eventually close `cmd_tx` and let the worker
        // exit, but we want to guarantee the thread has fully terminated
        // before the `Box<dyn Env>` it owns is gone (otherwise destructors
        // could race with Python interpreter finalization).
        let (dummy_tx, _) = std::sync::mpsc::channel();
        let closed_tx = std::mem::replace(&mut self.cmd_tx, dummy_tx);
        drop(closed_tx);
        if let Some(h) = self.handle.take() {
            let _ = h.join();
        }
    }
}

use std::sync::{Arc, Mutex};

struct WorkerPool {
    workers: Vec<Arc<Worker>>,
    slot_by_env_id: HashMap<usize, usize>,
    /// All workers send tagged outputs into this shared channel. Collection
    /// drains exactly `workers.len()` messages per `step`/`reset` and slots
    /// each one by `env_id`, so the first-done worker unblocks first and
    /// step latency becomes `max(worker_i)` rather than `sum(worker_i)`.
    /// The `Mutex` only exists to make `Receiver` satisfy `Sync`; there is
    /// no real contention (the pyclass is `unsendable`, so only one thread
    /// holds the pool at a time).
    result_rx: Arc<Mutex<std::sync::mpsc::Receiver<WorkerOutput>>>,
}

impl WorkerPool {
    fn new(envs: Vec<Box<dyn Env>>) -> Self {
        let (result_tx, result_rx) = std::sync::mpsc::channel();
        let workers: Vec<Arc<Worker>> = envs
            .into_iter()
            .enumerate()
            .map(|(i, e)| Arc::new(Worker::spawn(i, e, result_tx.clone())))
            .collect();
        let slot_by_env_id = build_slot_map(&workers);
        // Drop the original sender; workers hold the only remaining clones.
        // When every worker terminates, the receiver disconnects cleanly.
        drop(result_tx);
        Self {
            workers,
            slot_by_env_id,
            result_rx: Arc::new(Mutex::new(result_rx)),
        }
    }

    fn len(&self) -> usize {
        self.workers.len()
    }

    fn get(&self, index: usize) -> Self {
        let workers = vec![self.workers[index].clone()];
        Self {
            slot_by_env_id: build_slot_map(&workers),
            workers,
            result_rx: self.result_rx.clone(),
        }
    }

    fn slice(&self, start: usize, end: usize) -> Self {
        let workers = self.workers[start..end].to_vec();
        Self {
            slot_by_env_id: build_slot_map(&workers),
            workers,
            result_rx: self.result_rx.clone(),
        }
    }

    fn step(&self, actions: &[usize]) -> PyResult<Vec<EnvStepOutput>> {
        for (w, &action) in self.workers.iter().zip(actions.iter()) {
            w.step(action);
        }
        self.collect_results()
    }

    fn reset(&self) -> PyResult<Vec<EnvStepOutput>> {
        for w in &self.workers {
            w.reset();
        }
        self.collect_results()
    }

    fn collect_results(&self) -> PyResult<Vec<EnvStepOutput>> {
        let rx = self.result_rx.lock().expect("poisoned result receiver");
        let n = self.workers.len();

        // Drain N results in arrival order (first-done first).
        let mut by_slot: Vec<Option<WorkerOutput>> = (0..n).map(|_| None).collect();
        for _ in 0..n {
            let out = rx.recv().map_err(|_| worker_gone_error())?;
            let id = match &out {
                Ok(o) => o.env_id,
                Err(p) => p.env_id,
            };
            let slot = *self
                .slot_by_env_id
                .get(&id)
                .ok_or_else(stray_result_error)?;
            by_slot[slot] = Some(out);
        }

        // Reorder into slice-position order, surfacing any panic as PyErr.
        by_slot
            .into_iter()
            .map(|out| match out.ok_or_else(stray_result_error)? {
                Ok(o) => Ok(o),
                Err(p) => Err(worker_panicked_error(p.env_id, &p.msg)),
            })
            .collect()
    }

    /// Hand consumed obs buffers back to their originating worker so the
    /// worker's scratch pool keeps the capacity-preserving Vecs around.
    /// Call after you've finished reading an outputs batch.
    fn return_buffers(&self, outputs: Vec<EnvStepOutput>) {
        for (w, mut out) in self.workers.iter().zip(outputs) {
            w.return_buf(std::mem::take(&mut out.obs));
        }
    }
}

fn build_slot_map(workers: &[Arc<Worker>]) -> HashMap<usize, usize> {
    workers
        .iter()
        .enumerate()
        .map(|(slot, worker)| (worker.env_id, slot))
        .collect()
}

/// Extract a config value from a Python dict, falling back to `default` if the key
/// is missing. A type-mismatched value is reported as a `PyTypeError` instead of
/// being silently replaced with the default — misconfiguration should be loud.
fn get_config_or<T: for<'py> pyo3::FromPyObject<'py>>(
    config: &Bound<'_, PyDict>,
    key: &str,
    default: T,
) -> PyResult<T> {
    match config.get_item(key)? {
        None => Ok(default),
        Some(v) => v.extract(),
    }
}

/// Write a batch of env outputs into pre-allocated numpy arrays and info dicts.
fn write_step_outputs<'py>(
    py: Python<'py>,
    outputs: &[EnvStepOutput],
    obs_arr: &Bound<'py, numpy::PyArray2<f32>>,
    rewards_arr: &Bound<'py, numpy::PyArray1<f32>>,
    terminated_arr: &Bound<'py, numpy::PyArray1<bool>>,
    truncated_arr: &Bound<'py, numpy::PyArray1<bool>>,
) -> PyResult<Vec<Bound<'py, PyDict>>> {
    use numpy::PyArrayMethods;

    // SAFETY: We hold the GIL and these arrays were just allocated by us, so
    // no other Python thread can access them concurrently.
    let mut obs_rw = unsafe { obs_arr.as_array_mut() };
    let mut rew_rw = unsafe { rewards_arr.as_array_mut() };
    let mut term_rw = unsafe { terminated_arr.as_array_mut() };
    let mut trunc_rw = unsafe { truncated_arr.as_array_mut() };

    let mut infos = Vec::with_capacity(outputs.len());
    for (i, out) in outputs.iter().enumerate() {
        obs_rw
            .row_mut(i)
            .assign(&ndarray::ArrayView1::from(out.obs.as_slice()));
        rew_rw[i] = out.result.reward;
        term_rw[i] = out.result.terminated;
        trunc_rw[i] = false;
        let info = step_result_to_dict(py, &out.result)?;
        if let Some(ref final_obs) = out.final_obs {
            let fo = numpy::PyArray1::from_slice(py, final_obs.as_slice());
            info.set_item("final_observation", fo)?;
        }
        infos.push(info);
    }
    Ok(infos)
}

/// Extract a required value from a Python dict, returning `PyKeyError` if missing.
fn get_config_required<'py, T: pyo3::FromPyObject<'py>>(
    config: &Bound<'py, PyDict>,
    key: &str,
) -> PyResult<T> {
    config
        .get_item(key)?
        .ok_or_else(|| pyo3::exceptions::PyKeyError::new_err(key.to_string()))?
        .extract()
}

impl EnvConfig {
    fn from_dict(cfg: &Bound<'_, PyDict>) -> PyResult<Self> {
        Ok(Self {
            rom: get_config_required(cfg, "rom")?,
            actions: get_config_required(cfg, "actions")?,
            world: get_config_or::<Option<u8>>(cfg, "world", None)?,
            stage: get_config_or::<Option<u8>>(cfg, "stage", None)?,
            skip: get_config_or(cfg, "skip", 4)?,
            episodic_life: get_config_or(cfg, "episodic_life", true)?,
            custom_reward: get_config_or(cfg, "custom_reward", false)?,
            clip_rewards: get_config_or(cfg, "clip_rewards", true)?,
            frame_stack: get_config_or(cfg, "frame_stack", 4)?,
        })
    }
}

/// Vectorized SMB environment running N emulators across N worker threads.
///
/// Each config dict specifies one env with the same keys accepted by
/// `NesSmbEnv.__init__` (`rom`, `actions`, `world`, `stage`, `skip`,
/// `episodic_life`, `custom_reward`, `clip_rewards`, `frame_stack`).
/// Missing optional keys fall back to the `NesSmbEnv` defaults; wrong-
/// type values raise `TypeError`.
///
/// `step(actions)` auto-resets terminated sub-envs and exposes the
/// terminal observation under `info["final_observation"]`.
///
/// Slicing (`env[0:2]`, `env[-1]`) returns a view sharing the same
/// workers. Sub-envs must be stepped independently of the parent.
#[pyclass(name = "NesVecSmbEnv", unsendable)]
pub struct PyNesVecSmbEnv {
    pool: WorkerPool,
    num_envs: usize,
    obs_size: usize,
    num_actions: usize,
}

#[pymethods]
impl PyNesVecSmbEnv {
    /// Construct a vec env from a list of per-env config dicts.
    ///
    /// Raises:
    ///     ValueError: `configs` is empty.
    ///     KeyError: a config is missing a required key (`rom`, `actions`).
    ///     TypeError: a config value has the wrong type.
    ///     IOError: a ROM can't be loaded.
    #[new]
    #[pyo3(signature = (configs))]
    fn new(configs: Vec<Bound<'_, PyDict>>) -> PyResult<Self> {
        if configs.is_empty() {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "configs must not be empty",
            ));
        }
        let num_envs = configs.len();
        let mut envs: Vec<Box<dyn Env>> = Vec::with_capacity(num_envs);

        for cfg_dict in &configs {
            let cfg = EnvConfig::from_dict(cfg_dict)?;
            envs.push(build_env(cfg)?);
        }

        // Safe: `configs` is non-empty, so `envs` is too.
        let obs_size = envs[0].obs().len();
        let num_actions = envs[0].num_actions();
        for (i, e) in envs.iter().enumerate().skip(1) {
            if e.obs().len() != obs_size || e.num_actions() != num_actions {
                return Err(pyo3::exceptions::PyValueError::new_err(format!(
                    "env {i} has obs_size={} num_actions={}, but env 0 has obs_size={obs_size} num_actions={num_actions}",
                    e.obs().len(),
                    e.num_actions(),
                )));
            }
        }
        let pool = WorkerPool::new(envs);

        Ok(Self {
            pool,
            num_envs,
            obs_size,
            num_actions,
        })
    }

    fn reset<'py>(&mut self, py: Python<'py>) -> PyResult<(PyObject, PyObject)> {
        use numpy::{PyArray2, PyArrayMethods};

        let results = maybe_allow_threads(py, self.num_envs >= 2, || self.pool.reset())?;
        let obs_arr = PyArray2::<f32>::zeros(py, [self.num_envs, self.obs_size], false);
        let mut infos: Vec<Bound<'py, PyDict>> = Vec::with_capacity(self.num_envs);
        // Build infos first; return buffers unconditionally before surfacing
        // any error so a partial failure can't leak buffers out of the pool.
        let build_result: PyResult<()> = (|| {
            // SAFETY: Array just allocated by us; GIL held, no concurrent access.
            let mut obs_rw = unsafe { obs_arr.as_array_mut() };
            for (i, out) in results.iter().enumerate() {
                obs_rw
                    .row_mut(i)
                    .assign(&ndarray::ArrayView1::from(out.obs.as_slice()));
                infos.push(step_result_to_dict(py, &out.result)?);
            }
            Ok(())
        })();
        self.pool.return_buffers(results);
        build_result?;
        let info_list = pyo3::types::PyList::new(py, &infos)?;
        Ok((obs_arr.into_any().unbind(), info_list.into_any().unbind()))
    }

    fn step<'py>(
        &mut self,
        py: Python<'py>,
        actions: Vec<usize>,
    ) -> PyResult<(PyObject, PyObject, PyObject, PyObject, PyObject)> {
        use numpy::{PyArray1, PyArray2};

        if actions.len() != self.num_envs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "expected {} actions, got {}",
                self.num_envs,
                actions.len()
            )));
        }
        if let Some(&bad) = actions.iter().find(|&&a| a >= self.num_actions) {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "action {} out of range (expected 0..{})",
                bad, self.num_actions
            )));
        }
        let outputs = maybe_allow_threads(py, self.num_envs >= 2, || self.pool.step(&actions))?;
        let obs_arr = PyArray2::<f32>::zeros(py, [self.num_envs, self.obs_size], false);
        let rewards_arr = PyArray1::<f32>::zeros(py, self.num_envs, false);
        let terminated_arr = PyArray1::<bool>::zeros(py, self.num_envs, false);
        let truncated_arr = PyArray1::<bool>::zeros(py, self.num_envs, false);

        // Return buffers to the pool unconditionally — if `write_step_outputs`
        // fails partway, the already-consumed Vecs would otherwise leak out
        // of the worker's scratch pool for the rest of the process lifetime.
        let write_result = write_step_outputs(
            py,
            &outputs,
            &obs_arr,
            &rewards_arr,
            &terminated_arr,
            &truncated_arr,
        );
        self.pool.return_buffers(outputs);
        let infos = write_result?;

        let info_list = pyo3::types::PyList::new(py, &infos)?;
        Ok((
            obs_arr.into_any().unbind(),
            rewards_arr.into_any().unbind(),
            terminated_arr.into_any().unbind(),
            truncated_arr.into_any().unbind(),
            info_list.into_any().unbind(),
        ))
    }

    fn __getitem__(&self, key: &Bound<'_, pyo3::types::PyAny>) -> PyResult<Self> {
        if let Ok(slice) = key.downcast::<pyo3::types::PySlice>() {
            return self.get_slice(slice);
        }
        if let Ok(idx) = key.extract::<isize>() {
            return self.get_index(idx);
        }
        Err(pyo3::exceptions::PyTypeError::new_err(
            "NesVecSmbEnv indices must be integers or slices",
        ))
    }

    #[getter]
    fn num_envs(&self) -> usize {
        self.num_envs
    }
    #[getter]
    fn obs_size(&self) -> usize {
        self.obs_size
    }
}

impl PyNesVecSmbEnv {
    fn get_slice(&self, slice: &Bound<'_, pyo3::types::PySlice>) -> PyResult<Self> {
        let indices = slice.indices(self.num_envs as isize)?;
        if indices.step != 1 {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "only contiguous slices (step=1) are supported",
            ));
        }
        let (start, stop) = (indices.start as usize, indices.stop as usize);
        if start >= stop || stop > self.num_envs {
            return Err(pyo3::exceptions::PyValueError::new_err(format!(
                "slice [{}:{}] out of range for {} envs",
                start, stop, self.num_envs
            )));
        }
        Ok(self.slice(self.pool.slice(start, stop)))
    }

    fn get_index(&self, idx: isize) -> PyResult<Self> {
        let n = self.num_envs as isize;
        let i = if idx < 0 {
            (n + idx) as usize
        } else {
            idx as usize
        };
        if i >= self.num_envs {
            return Err(pyo3::exceptions::PyIndexError::new_err(format!(
                "index {} out of range for {} envs",
                idx, self.num_envs
            )));
        }
        Ok(self.slice(self.pool.get(i)))
    }

    fn slice(&self, pool: WorkerPool) -> Self {
        let num_envs = pool.len();
        Self {
            pool,
            num_envs,
            obs_size: self.obs_size,
            num_actions: self.num_actions,
        }
    }
}
