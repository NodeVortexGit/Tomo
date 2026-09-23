//! Seeing the screen: screenshots the model can look at.
//!
//! The AI asks for one through its `look_at_screen` tool (ai.rs). Capture
//! shells out to whichever screenshot tool the desktop has — `grim` on
//! wlroots compositors (Hyprland, Sway), `spectacle` on KDE,
//! `gnome-screenshot` on GNOME, `maim` / `scrot` / ImageMagick `import` on X11
//! — then fits the image for the model: at most [`MAX_EDGE`] px on its long
//! edge (1080p-class, the sweet spot between detail and image tokens), as JPEG.

use std::io::Cursor;
use std::process::Stdio;

use anyhow::{anyhow, Result};
use image::codecs::jpeg::JpegEncoder;
use image::imageops::FilterType;
use image::{ImageFormat, ImageReader};
use tokio::process::Command;

/// Longest edge sent to the model, px.
const MAX_EDGE: u32 = 1920;
const JPEG_QUALITY: u8 = 80;

/// A JPEG ready to send, and its size.
pub struct Screenshot {
    pub jpeg: Vec<u8>,
    pub width: u32,
    pub height: u32,
}

/// Capture the whole screen.
pub async fn capture() -> Result<Screenshot> {
    let raw = grab().await?;
    tokio::task::spawn_blocking(move || fit_for_model(&raw)).await?
}

/// Run the first screenshot tool that works and return its image.
async fn grab() -> Result<Vec<u8>> {
    let file = std::env::temp_dir().join(format!("tomo-screen-{}.png", std::process::id()));
    let path = file.to_string_lossy().to_string();
    // (program, arguments, whether it writes the image to stdout or to `file`)
    let tools: [(&str, Vec<&str>, bool); 6] = [
        ("grim", vec!["-t", "jpeg", "-q", "80", "-"], true),
        ("spectacle", vec!["-b", "-n", "-f", "-o", &path], false),
        ("gnome-screenshot", vec!["-f", &path], false),
        ("maim", vec![], true),
        ("scrot", vec!["-o", &path], false),
        ("import", vec!["-window", "root", "png:-"], true),
    ];
    for (program, args, to_stdout) in tools {
        // An error here means it isn't installed; a failure means it doesn't
        // work on this desktop (e.g. grim on X11). Either way, try the next.
        let Ok(output) = Command::new(program)
            .args(&args)
            .stdin(Stdio::null())
            .stderr(Stdio::null())
            .output()
            .await
        else {
            continue;
        };
        if !output.status.success() {
            continue;
        }
        let image = if to_stdout {
            output.stdout
        } else {
            let read = tokio::fs::read(&file).await;
            let _ = tokio::fs::remove_file(&file).await;
            read.unwrap_or_default()
        };
        if !image.is_empty() {
            return Ok(image);
        }
    }
    Err(anyhow!(
        "no screenshot tool worked (install grim on Wayland, or maim or scrot on X11)"
    ))
}

/// Scale to at most [`MAX_EDGE`] and encode as JPEG. A JPEG that already fits
/// (grim's usual output) goes through untouched.
fn fit_for_model(raw: &[u8]) -> Result<Screenshot> {
    let reader = ImageReader::new(Cursor::new(raw)).with_guessed_format()?;
    let format = reader.format();
    let (width, height) = reader.into_dimensions()?;
    if format == Some(ImageFormat::Jpeg) && width.max(height) <= MAX_EDGE {
        return Ok(Screenshot {
            jpeg: raw.to_vec(),
            width,
            height,
        });
    }

    let mut image = image::load_from_memory(raw)?;
    if width.max(height) > MAX_EDGE {
        image = image.resize(MAX_EDGE, MAX_EDGE, FilterType::Triangle);
    }
    let rgb = image.to_rgb8();
    let mut jpeg = Vec::new();
    JpegEncoder::new_with_quality(&mut jpeg, JPEG_QUALITY).encode_image(&rgb)?;
    Ok(Screenshot {
        jpeg,
        width: rgb.width(),
        height: rgb.height(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use image::{DynamicImage, RgbImage};

    fn encoded(width: u32, height: u32, format: ImageFormat) -> Vec<u8> {
        let mut bytes = Vec::new();
        DynamicImage::ImageRgb8(RgbImage::new(width, height))
            .write_to(&mut Cursor::new(&mut bytes), format)
            .unwrap();
        bytes
    }

    #[test]
    fn a_large_screenshot_is_scaled_down_to_jpeg() {
        let shot = fit_for_model(&encoded(3840, 2160, ImageFormat::Png)).unwrap();
        assert_eq!((shot.width, shot.height), (1920, 1080));
        assert_eq!(&shot.jpeg[..2], &[0xFF, 0xD8], "JPEG magic");
    }

    #[test]
    fn a_small_png_is_converted_but_keeps_its_size() {
        let shot = fit_for_model(&encoded(1366, 768, ImageFormat::Png)).unwrap();
        assert_eq!((shot.width, shot.height), (1366, 768));
        assert_eq!(&shot.jpeg[..2], &[0xFF, 0xD8]);
    }

    #[test]
    fn a_fitting_jpeg_goes_through_untouched() {
        let jpeg = encoded(1920, 1080, ImageFormat::Jpeg);
        let shot = fit_for_model(&jpeg).unwrap();
        assert_eq!(shot.jpeg, jpeg);
    }
}
