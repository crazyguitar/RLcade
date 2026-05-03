"""Super Mario Bros gymnasium environment - thin Python wrapper over Rust NesSmbEnv."""

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from nes import NesSmbEnv, NesVecSmbEnv

# Button masks

A = 0x01
B = 0x02
SELECT = 0x04
START = 0x08
UP = 0x10
DOWN = 0x20
LEFT = 0x40
RIGHT = 0x80

# Predefined action spaces (each entry is a joypad bitmask)

# fmt: off
RIGHT_ONLY = [
    0,              # NOOP
    RIGHT,          # Right
    RIGHT | A,      # Right + A (jump right)
    RIGHT | B,      # Right + B (run right)
    RIGHT | A | B,  # Right + A + B (run + jump right)
    A,              # A (jump)
]

SIMPLE_MOVEMENT = [
    0,              # NOOP
    RIGHT,          # Right
    RIGHT | A,      # Right + A
    RIGHT | B,      # Right + B
    RIGHT | A | B,  # Right + A + B
    A,              # A
    LEFT,           # Left
]

COMPLEX_MOVEMENT = [
    0,              # NOOP
    RIGHT,          # Right
    RIGHT | A,      # Right + A
    RIGHT | B,      # Right + B
    RIGHT | A | B,  # Right + A + B
    A,              # A
    LEFT,           # Left
    LEFT | A,       # Left + A
    LEFT | B,       # Left + B
    LEFT | A | B,   # Left + A + B
    DOWN,           # Down
    UP,             # Up
]
# fmt: on

ACTIONS = {
    "right": RIGHT_ONLY,
    "simple": SIMPLE_MOVEMENT,
    "complex": COMPLEX_MOVEMENT,
}

RENDER_HUMAN = "human"
RENDER_RGB_ARRAY = "rgb_array"

# Observation shape after warp + frame stack: (K, 84, 84) CHW float32
OBS_H = 84
OBS_W = 84


class SuperMarioBrosEnv(gym.Env):
    """Gymnasium wrapper over the Rust NesSmbEnv.

    The full pipeline runs in Rust:
    MaxAndSkip → reward/done/info → RAM hacks → EpisodicLife →
    CustomReward → Grayscale → Resize → Scale → ClipReward → FrameStack

    Observations are numpy arrays (C, H, W) float32 in [0, 1].
    """

    metadata = {"render_modes": [RENDER_HUMAN, RENDER_RGB_ARRAY], "render_fps": 60}

    def __init__(self, config):
        super().__init__()
        actions = config.actions if config.actions is not None else list(SIMPLE_MOVEMENT)
        if isinstance(actions, str):
            actions = list(ACTIONS[actions])

        # Input validation
        if config.stage is not None and config.world is None:
            raise ValueError("Cannot specify stage without world. Use world=w, stage=s or world=w, stage=None")

        self.render_mode = config.render_mode
        frame_stack = getattr(config, "frame_stack", 4)

        # Create NesSmbEnv
        self._env = NesSmbEnv(
            rom=config.rom_path,
            actions=actions,
            world=config.world,
            stage=config.stage,
            skip=getattr(config, "skip", 4),
            episodic_life=getattr(config, "episodic_life", True),
            custom_reward=getattr(config, "custom_reward", False),
            clip_rewards=getattr(config, "clip_rewards", True),
            frame_stack=frame_stack,
        )

        self.action_space = spaces.Discrete(len(actions))
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(frame_stack, OBS_H, OBS_W),
            dtype=np.float32,
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        obs, info = self._env.reset()
        return obs.reshape(self.observation_space.shape), info

    def step(self, action):
        # The Rust step follows Gym semantics: one agent decision step,
        # including frame-skip, with one Python boundary crossing.
        obs, reward, terminated, truncated, info = self._env.step(action)
        return (
            obs.reshape(self.observation_space.shape),
            reward,
            terminated,
            truncated,
            info,
        )

    def render(self):
        if self.render_mode == RENDER_HUMAN:
            self._env.render()
            return None
        if self.render_mode == RENDER_RGB_ARRAY:
            buf = self._env.screen().bytearray()
            return np.frombuffer(buf, dtype=np.uint8).reshape(self._env.screen_height, self._env.screen_width, 3)
        return None

    def poll_quit(self) -> bool:
        if self.render_mode == RENDER_HUMAN:
            return self._env.poll_quit()
        return False

    def close(self):
        pass

    @property
    def _life(self) -> int:
        """Expose life for EpisodicLifeEnv compatibility."""
        return self._env.life


class SuperMarioBrosVecEnv:
    """In-process vectorized SMB environment — no subprocess IPC.

    All N emulators run in a single process via Rust, avoiding
    AsyncVectorEnv's per-step serialization overhead.
    """

    def __init__(self, configs: list[dict]):
        for cfg in configs:
            actions = cfg.get("actions", "simple")
            if isinstance(actions, str):
                cfg["actions"] = list(ACTIONS[actions])

        self._env = NesVecSmbEnv(configs)
        self.num_envs = self._env.num_envs
        frame_stack = configs[0].get("frame_stack", 4) if configs else 4
        single_obs_shape = (frame_stack, OBS_H, OBS_W)

        n_actions = len(configs[0]["actions"]) if configs else 0
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(self.num_envs, *single_obs_shape),
            dtype=np.float32,
        )
        self.single_observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=single_obs_shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Discrete(n_actions)
        self.single_action_space = spaces.Discrete(n_actions)
        self._obs_shape = single_obs_shape

    def reset(self, **kwargs):
        obs_flat, infos = self._env.reset()
        return np.asarray(obs_flat).reshape(self.num_envs, *self._obs_shape), infos

    def step(self, actions):
        action_list = actions.tolist() if hasattr(actions, "tolist") else list(actions)
        obs_flat, rewards, terminated, truncated, infos = self._env.step(action_list)
        obs = np.asarray(obs_flat).reshape(self.num_envs, *self._obs_shape)
        return obs, np.asarray(rewards), np.asarray(terminated), np.asarray(truncated), infos

    def __getitem__(self, key):
        env = self._env[key]
        new = SuperMarioBrosVecEnv.__new__(SuperMarioBrosVecEnv)
        new._env = env
        new.num_envs = env.num_envs
        new._obs_shape = self._obs_shape
        new.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(new.num_envs, *self._obs_shape),
            dtype=np.float32,
        )
        new.single_observation_space = self.single_observation_space
        new.action_space = self.action_space
        new.single_action_space = self.single_action_space
        return new

    def close(self):
        pass
