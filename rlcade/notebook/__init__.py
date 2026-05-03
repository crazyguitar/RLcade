"""Jupyter notebook display helpers for NES emulation.

Architecture (display_wasm):

    Jupyter Notebook                    Browser (IFrame)
    +--------------+                   +----------------------------+
    | display_wasm |                   |                            |
    |   ("rom")    |                   |  1. Load nes.js + nes.wasm |
    |      |       |    HTTP serve     |  2. fetch("game.nes")      |
    |      v       | <---------------- |  3. WasmNes(rom_bytes)     |
    |  HTTP Server | ----------------> |         |                  |
    |  localhost   |   pkg/ + ROM      |         v                  |
    +--------------+                   |  +-------------+  Keyboard |
          |                            |  | WASM        | <---------|
     only serves                       |  | Emulator    |           |
     files, then                       |  | (Rust->wasm)|           |
     does nothing                      |  | Cpu,Ppu,..  |           |
                                       |  +------+------+           |
                                       |         | screen_rgb()     |
                                       |         v                  |
                                       |     <canvas>               |
                                       |      60fps                 |
                                       +----------------------------+

    Python serves the files once. After that, everything runs in the
    browser: the ROM is loaded into the WASM emulator's memory, and
    the emulate-render loop runs entirely client-side.
"""

import argparse
import io
import threading
import time
from dataclasses import dataclass, field
from functools import partial
from http.server import HTTPServer, SimpleHTTPRequestHandler
from pathlib import Path

from IPython.display import IFrame, display

from rlcade.envs import SuperMarioBrosConfig, SuperMarioBrosEnv

_INDEX_HTML = """\
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<title>NES Emulator</title>
<style>
  body {
    background: #1a1a2e;
    display: flex;
    flex-direction: column;
    align-items: center;
    justify-content: center;
    height: 100vh;
    margin: 0;
    font-family: monospace;
    color: #ccc;
  }
  canvas {
    width: 512px;
    height: 480px;
    image-rendering: pixelated;
    border: 2px solid #444;
  }
  .controls {
    font-size: 12px;
    margin-top: 8px;
  }
</style>
</head>
<body>
<canvas id="screen" width="256" height="240"></canvas>
<p class="controls">Arrow keys = D-pad, J = A, K = B, M = Start, N = Select</p>

<script type="module">
import init, { WasmNes } from './pkg/nes.js';
const NES_FPS = 60;
const FRAME_MS = 1000 / NES_FPS;
const KEYS = {
  ArrowUp: 0x10, ArrowDown: 0x20, ArrowLeft: 0x40, ArrowRight: 0x80,
  KeyJ: 0x01, KeyK: 0x02, KeyN: 0x04, KeyM: 0x08,
};

let joypad = 0;
document.addEventListener('keydown', e => { if (KEYS[e.code]) joypad |= KEYS[e.code]; });
document.addEventListener('keyup',   e => { if (KEYS[e.code]) joypad &= ~KEYS[e.code]; });
await init();
const romPath = new URLSearchParams(location.search).get('rom') || 'game.nes';
const romData = new Uint8Array(await fetch(romPath).then(r => r.arrayBuffer()));
const nes = new WasmNes(romData);
const ctx = document.getElementById('screen').getContext('2d');
const frame = ctx.createImageData(256, 240);
// Render loop (60 fps cap)
function rgbToImageData(rgb, dst) {
  for (let i = 0, j = 0; i < rgb.length; i += 3, j += 4) {
    dst[j] = rgb[i]; dst[j + 1] = rgb[i + 1];
    dst[j + 2] = rgb[i + 2]; dst[j + 3] = 255;
  }
}
let lastTime = 0;
function tick(now) {
  requestAnimationFrame(tick);
  if (now - lastTime < FRAME_MS) return;
  lastTime = now;
  nes.step_frame(joypad, 0);
  rgbToImageData(nes.screen_rgb(), frame.data);
  ctx.putImageData(frame, 0, 0);
}
requestAnimationFrame(tick);
</script>
</body>
</html>
"""


SCREEN_WIDTH = 512
SCREEN_HEIGHT = 480
DEFAULT_PORT = 8080
NES_ENV = "rlcade/SuperMarioBros-v0"


@dataclass
class AgentConfig:
    """Configuration for agent playback display."""

    rom: str
    checkpoint: str
    agent: str = "ppo"
    actions: str = "complex"
    world: int = 1
    stage: int = 1
    encoder: str = "cnn"
    device: str = "cpu"
    num_steps: int = 1000
    fps: int = 30
    # LSTM defaults
    lstm_hidden: int = 256
    lstm_layers: int = 1
    # ICM defaults
    icm: bool = False
    icm_coef: float = 0.01
    icm_feature_dim: int = 256


# Project helpers


def _find_project_root():
    root = Path(__file__).resolve().parent.parent.parent
    if not (root / "pkg").exists():
        raise FileNotFoundError(f"WASM package not found at {root / 'pkg'}. Run 'make wasm' first.")
    return root


def _ensure_index_html(root):
    index = root / "index.html"
    if not index.exists():
        index.write_text(_INDEX_HTML)


_servers: dict[tuple[str, int], HTTPServer] = {}


def _start_server(root, port):
    _ensure_index_html(root)
    key = (str(root), port)
    existing = _servers.get(key)
    if existing is not None:
        return existing
    handler = partial(SimpleHTTPRequestHandler, directory=str(root))
    server = HTTPServer(("localhost", port), handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    _servers[key] = server
    return server


# Agent helpers


def _create_env(cfg):
    config = SuperMarioBrosConfig(
        rom_path=cfg.rom,
        actions=cfg.actions,
        world=cfg.world,
        stage=cfg.stage,
        render_mode="rgb_array",
    )
    return SuperMarioBrosEnv(config)


def _load_trained_agent(cfg, env):
    from rlcade.agent import load_agent
    from rlcade.utils import resolve_device

    args = argparse.Namespace(
        agent=cfg.agent,
        checkpoint=cfg.checkpoint,
        device=resolve_device(cfg.device),
        encoder=cfg.encoder,
        obs_shape=env.observation_space.shape,
        n_actions=env.action_space.n,
        rom=cfg.rom,
        env=NES_ENV,
        actions=cfg.actions,
        world=cfg.world,
        stage=cfg.stage,
        render_mode="rgb_array",
        custom_reward=False,
        lstm_hidden=cfg.lstm_hidden,
        lstm_layers=cfg.lstm_layers,
        icm=cfg.icm,
        icm_coef=cfg.icm_coef,
        icm_feature_dim=cfg.icm_feature_dim,
    )
    return load_agent(cfg.agent, args, env), args.device


def _render_frame(env):
    """Render the current screen as PNG bytes."""
    from PIL import Image

    rgb = env.render()
    img = Image.fromarray(rgb)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def _run_agent_loop(env, agent, device, canvas, info_label, cfg):
    """Step the agent and stream frames to the widget."""
    import torch

    obs, _ = env.reset()
    frame_delay = 1.0 / cfg.fps
    episode = 0

    for _ in range(cfg.num_steps):
        frame_start = time.time()

        obs_tensor = torch.from_numpy(obs).float().to(device)
        action, _, _ = agent.get_action(obs_tensor)
        obs, _, terminated, truncated, info = env.step(action.item())
        canvas.value = _render_frame(env)

        if terminated or truncated:
            episode += 1
            info_label.value = f"Episode {episode} | Score: {info['score']} | x: {info['x_pos']}"
            obs, _ = env.reset()

        elapsed = time.time() - frame_start
        if elapsed < frame_delay:
            time.sleep(frame_delay - elapsed)


# Public API


def display_wasm(rom, *, port=DEFAULT_PORT, width=530, height=510):
    """Display the NES emulator in a Jupyter notebook via WASM.

    Runs the emulator in the browser with keyboard input.
    """
    root = _find_project_root()
    rom_path = Path(rom).resolve()
    if not rom_path.exists():
        raise FileNotFoundError(f"ROM not found: {rom_path}")

    _start_server(root, port)
    relative_rom = rom_path.relative_to(root)
    display(IFrame(f"http://localhost:{port}/index.html?rom={relative_rom}", width=width, height=height))


def display_agent(rom, checkpoint, **kwargs):
    """Display a trained agent playing in a Jupyter notebook.

    Runs the agent in Python and streams frames via ipywidgets.
    """
    import ipywidgets as widgets

    cfg = AgentConfig(rom=rom, checkpoint=checkpoint, **kwargs)
    env = _create_env(cfg)
    agent, device = _load_trained_agent(cfg, env)

    canvas = widgets.Image(format="png", width=SCREEN_WIDTH, height=SCREEN_HEIGHT)
    info_label = widgets.Label(value="")
    display(widgets.VBox([canvas, info_label]))

    try:
        _run_agent_loop(env, agent, device, canvas, info_label, cfg)
    finally:
        env.close()
