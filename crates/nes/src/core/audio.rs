use super::apu::{AUDIO_BUFFER_SAMPLES, SAMPLE_RATE};
use sdl2::audio::{AudioCallback, AudioDevice, AudioSpecDesired};
use std::collections::VecDeque;
use std::sync::{Arc, Mutex};

/// SDL2 audio callback that drains samples from a shared ring buffer.
struct AudioOutput {
    buffer: Arc<Mutex<VecDeque<f32>>>,
    last_sample: f32,
}

impl AudioCallback for AudioOutput {
    type Channel = f32;

    fn callback(&mut self, out: &mut [f32]) {
        let mut buffer = self.buffer.lock().unwrap();
        for sample in out.iter_mut() {
            if let Some(s) = buffer.pop_front() {
                self.last_sample = s;
            }
            *sample = self.last_sample;
        }
    }
}

/// Manages SDL2 audio playback with a shared sample ring buffer.
pub struct Audio {
    device: AudioDevice<AudioOutput>,
    buffer: Arc<Mutex<VecDeque<f32>>>,
    started: bool,
}

impl Audio {
    /// Initialize SDL2 audio playback.
    pub fn new(sdl: &sdl2::Sdl) -> Self {
        let buffer: Arc<Mutex<VecDeque<f32>>> = Arc::new(Mutex::new(VecDeque::new()));
        let audio = sdl.audio().expect("Failed to initialize SDL2 audio");
        let spec = AudioSpecDesired {
            freq: Some(SAMPLE_RATE as i32),
            channels: Some(1),
            samples: Some(AUDIO_BUFFER_SAMPLES),
        };
        let cb_buffer = Arc::clone(&buffer);
        let device = audio
            .open_playback(None, &spec, |_| AudioOutput {
                buffer: cb_buffer,
                last_sample: 0.0,
            })
            .expect("Failed to open audio device");
        Self {
            device,
            buffer,
            started: false,
        }
    }

    /// Push samples, start playback when buffered, and trim overflow.
    pub fn flush(&mut self, samples: &[f32]) {
        if let Ok(mut buf) = self.buffer.lock() {
            buf.extend(samples.iter());

            if !self.started && buf.len() >= AUDIO_BUFFER_SAMPLES as usize * 2 {
                self.device.resume();
                self.started = true;
            }

            const MAX_BUFFER_SIZE: usize = SAMPLE_RATE as usize / 10;
            if buf.len() > MAX_BUFFER_SIZE {
                let excess = buf.len() - MAX_BUFFER_SIZE;
                drop(buf.drain(..excess));
            }
        }
    }

    /// Reset audio state (e.g. after emulator reset).
    pub fn reset(&mut self) {
        self.started = false;
    }
}
