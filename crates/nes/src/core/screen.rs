#[cfg(feature = "sdl2-backend")]
use sdl2::pixels::PixelFormatEnum;
#[cfg(feature = "sdl2-backend")]
use sdl2::render::{Canvas, Texture};
#[cfg(feature = "sdl2-backend")]
use sdl2::video::Window;

const WINDOW_SCALE: u32 = 2;
const RGB_BYTES_PER_PIXEL: usize = 3;

/// Manages the SDL2 window and rendering.
#[cfg(feature = "sdl2-backend")]
pub struct Screen {
    canvas: Canvas<Window>,
    width: usize,
    height: usize,
}

#[cfg(feature = "sdl2-backend")]
impl Screen {
    /// Create a scaled SDL2 window with an accelerated canvas.
    pub fn new(sdl: &sdl2::Sdl, width: usize, height: usize) -> Self {
        let video = sdl.video().expect("Failed to initialize SDL2 video");
        let window = video
            .window(
                "NES Emulator",
                width as u32 * WINDOW_SCALE,
                height as u32 * WINDOW_SCALE,
            )
            .position_centered()
            .build()
            .expect("Failed to create window");
        let canvas = window
            .into_canvas()
            .accelerated()
            .build()
            .expect("Failed to create canvas");
        Self {
            canvas,
            width,
            height,
        }
    }

    /// Present the given pixel buffer (u32 per pixel) to the window.
    pub fn present(&mut self, buffer: &[u32]) {
        let texture_creator = self.canvas.texture_creator();
        let mut texture = texture_creator
            .create_texture_streaming(
                PixelFormatEnum::RGB24,
                self.width as u32,
                self.height as u32,
            )
            .expect("Failed to create texture");
        self.blit(&mut texture, buffer);
        self.canvas.clear();
        self.canvas
            .copy(&texture, None, None)
            .expect("Failed to copy texture");
        self.canvas.present();
    }

    /// Convert pixel buffer into RGB24 texture data.
    #[inline]
    fn blit(&self, texture: &mut Texture, buffer: &[u32]) {
        let width = self.width;
        let height = self.height;
        texture
            .with_lock(None, |pixels: &mut [u8], pitch: usize| {
                for y in 0..height {
                    let row_buf = &buffer[y * width..(y * width + width)];
                    let row_pix = &mut pixels[y * pitch..y * pitch + width * RGB_BYTES_PER_PIXEL];
                    rgb_unpack(row_buf, row_pix);
                }
            })
            .expect("Failed to update texture");
    }
}

/// Return the given pixel buffer as an RGB byte array `[H * W * 3]`.
///
/// Available on all targets (no SDL2 dependency).
#[inline]
pub fn to_rgb(buffer: &[u32], width: usize, height: usize) -> Vec<u8> {
    let mut rgb = vec![0u8; height * width * RGB_BYTES_PER_PIXEL];
    rgb_unpack(buffer, &mut rgb);
    rgb
}

/// Unpack packed u32 pixels (0x00RRGGBB) into interleaved R, G, B bytes.
///
/// Written as a tight loop over contiguous slices so LLVM can auto-vectorize
/// with SSE/AVX on x86_64 and NEON on AArch64.
#[inline]
pub fn rgb_unpack(src: &[u32], dst: &mut [u8]) {
    debug_assert!(dst.len() >= src.len() * RGB_BYTES_PER_PIXEL);
    for (i, &pixel) in src.iter().enumerate() {
        let offset = i * RGB_BYTES_PER_PIXEL;
        dst[offset] = (pixel >> 16) as u8;
        dst[offset + 1] = (pixel >> 8) as u8;
        dst[offset + 2] = pixel as u8;
    }
}
