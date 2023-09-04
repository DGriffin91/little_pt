use std::sync::Once;

use bytemuck::cast_slice;
use glam::{vec3a, Mat3A, UVec3, Vec3, Vec3A, Vec3Swizzles};
use shared_exponent_formats::rgb9e5;

use crate::d3_image::Image;

fn rgb_to_ycbcr(col: Vec3A) -> Vec3A {
    let m = Mat3A {
        x_axis: vec3a(0.2126, -0.1146, 0.5),
        y_axis: vec3a(0.7152, -0.3854, -0.4542),
        z_axis: vec3a(0.0722, 0.5, -0.0458),
    };

    m * col
}

fn tonemap_curve(v: f32) -> f32 {
    1.0 - (-v).exp()
}

fn tonemap_curve3(v: Vec3A) -> Vec3A {
    1.0 - (-v).exp()
}

fn tonemapping_luminance(col: Vec3A) -> f32 {
    // Replace this with your actual implementation for calculating luminance.
    // This is just a placeholder.
    col.dot(vec3a(0.2126, 0.7152, 0.0722))
}

pub fn somewhat_boring_display_transform(col: Vec3A) -> Vec3A {
    let mut col = col;
    let ycbcr = rgb_to_ycbcr(col);

    let bt = tonemap_curve(ycbcr.yz().length() * 2.4);
    let mut desat = (bt - 0.7) * 0.8;
    desat *= desat;

    let desat_col = col.lerp(ycbcr.xxx(), desat);

    let tm_luma = tonemap_curve(ycbcr.x);
    let tm0 = col * tm_luma / tonemapping_luminance(col).max(1e-5);
    let final_mult = 0.97;
    let tm1 = tonemap_curve3(desat_col);

    col = tm0.lerp(tm1, bt * bt);

    col * final_mult
}

// https://github.com/h3r2tic/tony-mc-mapface

pub fn tony_mc_mapface(stimulus: Vec3A) -> Vec3A {
    // Apply a non-linear transform that the LUT is encoded with.
    let encoded = stimulus / (stimulus + 1.0);

    // Align the encoded range to texel centers.
    let lut_dims = 48.0;
    let uv = encoded * ((lut_dims - 1.0) / lut_dims) + 0.5 / lut_dims;

    return get_tony_data().trilinear_sample(uv.into()).into();
}

static mut TONY_DATA: Option<Image> = None;
static START: Once = Once::new();

pub fn get_tony_data() -> &'static Image {
    START.call_once(|| {
        let byte_data: &[u8] = include_bytes!("tony_mc_mapface.dds");
        let dds = ddsfile::Dds::read(byte_data).unwrap();
        let mut rgb = Vec::new();
        for layer in 0..dds.get_num_array_layers() {
            let color_data = cast_slice::<u8, u32>(dds.get_data(layer).unwrap());
            for v in color_data {
                rgb.push(Vec3::from(rgb9e5::rgb9e5_to_vec3(*v)));
            }
        }

        unsafe {
            TONY_DATA = Some(Image {
                data: rgb,
                size: UVec3::splat(48),
            });
        }
    });

    unsafe { &TONY_DATA.as_ref().unwrap() }
}

pub mod save_tony_as_bytes {
    use glam::vec3;
    use image::{io::Reader as ImageReader, DynamicImage, ImageFormat};

    use crate::sampling::powsafe;

    use super::tony_mc_mapface;

    pub fn test_tony2() {
        let img = ImageReader::open("dragonscene_ap0_v01_1001.exr")
            .unwrap()
            .decode()
            .unwrap();

        let mut rgb = img.into_rgb32f();

        for rgb in rgb.pixels_mut() {
            let proc = tony_mc_mapface(vec3(rgb.0[0], rgb.0[1], rgb.0[2]).into()).into();
            //let proc =
            //    somewhat_boring_display_transform(vec3(rgb.0[0], rgb.0[1], rgb.0[2]).into()).into();
            rgb.0 = (powsafe(proc, 1.0 / 2.2)).into();
        }

        DynamicImage::from(rgb)
            .into_rgb8()
            .save_with_format("dragonscene_ap0_v01_1001_tony.png", ImageFormat::Png)
            .unwrap();
    }
}
