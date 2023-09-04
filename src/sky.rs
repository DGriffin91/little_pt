// Ported from:
// https://github.com/EmbarkStudios/rust-gpu/blob/main/examples/shaders/sky-shader/src/lib.rs
// https://github.com/Tw1ddle/Sky-Shader/tree/master

use core::f32::consts::PI;
use glam::{vec3a, Vec3A};

use crate::sampling::smoothstep;

const DEPOLARIZATION_FACTOR: f32 = 0.02;
const MIE_COEFFICIENT: f32 = 0.005;
const MIE_DIRECTIONAL_G: f32 = 0.82;
const MIE_K_COEFFICIENT: Vec3A = vec3a(0.686, 0.678, 0.666);
const MIE_V: f32 = 3.936;
const MIE_ZENITH_LENGTH: f32 = 34000.0;
const NUM_MOLECULES: f32 = 2.542e25f32;
const PRIMARIES: Vec3A = vec3a(6.8e-7f32, 5.5e-7f32, 4.5e-7f32);
const RAYLEIGH: f32 = 2.28;
const RAYLEIGH_ZENITH_LENGTH: f32 = 8400.0;
const REFRACTIVE_INDEX: f32 = 1.0003;
pub const SUN_ANGULAR_DIAMETER_DEGREES: f32 = 0.0093333;
const SUN_INTENSITY_FACTOR: f32 = 500.0;
const SUN_INTENSITY_FALLOFF_STEEPNESS: f32 = 1.5;
const TURBIDITY: f32 = 4.7;

fn total_rayleigh(lambda: Vec3A) -> Vec3A {
    (8.0 * PI.powf(3.0)
        * (REFRACTIVE_INDEX.powf(2.0) - 1.0).powf(2.0)
        * (6.0 + 3.0 * DEPOLARIZATION_FACTOR))
        / (3.0 * NUM_MOLECULES * lambda.powf(4.0) * (6.0 - 7.0 * DEPOLARIZATION_FACTOR))
}

fn total_mie(lambda: Vec3A, k: Vec3A, t: f32) -> Vec3A {
    let c = 0.2 * t * 10e-18;
    0.434 * c * PI * ((2.0 * PI) / lambda).powf(MIE_V - 2.0) * k
}

fn rayleigh_phase(cos_theta: f32) -> f32 {
    (3.0 / (16.0 * PI)) * (1.0 + cos_theta.powf(2.0))
}

fn henyey_greenstein_phase(cos_theta: f32, g: f32) -> f32 {
    (1.0 / (4.0 * PI)) * ((1.0 - g.powf(2.0)) / (1.0 - 2.0 * g * cos_theta + g.powf(2.0)).powf(1.5))
}

fn sun_intensity(zenith_angle_cos: f32) -> f32 {
    let cutoff_angle = PI / 1.95; // Earth shadow hack
    SUN_INTENSITY_FACTOR
        * 0.0f32.max(
            1.0 - (-((cutoff_angle - zenith_angle_cos.acos()) / SUN_INTENSITY_FALLOFF_STEEPNESS))
                .exp(),
        )
}

pub fn sky(dir: Vec3A, sun_position: Vec3A) -> Vec3A {
    let up = vec3a(0.0, 1.0, 0.0);
    let sunfade = 1.0 - (1.0 - (sun_position.y / 450000.0).clamp(0.0, 1.0).exp());
    let rayleigh_coefficient = RAYLEIGH - (1.0 * (1.0 - sunfade));
    let beta_r = total_rayleigh(PRIMARIES) * rayleigh_coefficient;

    // Mie coefficient
    let beta_m = total_mie(PRIMARIES, MIE_K_COEFFICIENT, TURBIDITY) * MIE_COEFFICIENT;

    // Optical length, cutoff angle at 90 to avoid singularity
    let zenith_angle = up.dot(dir).max(0.0).acos();
    let denom = (zenith_angle).cos() + 0.15 * (93.885 - ((zenith_angle * 180.0) / PI)).powf(-1.253);

    let s_r = RAYLEIGH_ZENITH_LENGTH / denom;
    let s_m = MIE_ZENITH_LENGTH / denom;

    // Combined extinction factor
    let fex = (-(beta_r * s_r + beta_m * s_m)).exp();

    // In-scattering
    let sun_direction = sun_position.normalize();
    let cos_theta = dir.dot(sun_direction);
    let beta_r_theta = beta_r * rayleigh_phase(cos_theta * 0.5 + 0.5);

    let beta_m_theta = beta_m * henyey_greenstein_phase(cos_theta, MIE_DIRECTIONAL_G);
    let sun_e = sun_intensity(sun_direction.dot(up));

    let mut lin =
        (sun_e * ((beta_r_theta + beta_m_theta) / (beta_r + beta_m)) * (Vec3A::splat(1.0) - fex))
            .powf(1.5);

    lin *= Vec3A::splat(1.0).lerp(
        (sun_e * ((beta_r_theta + beta_m_theta) / (beta_r + beta_m)) * fex).powf(0.5),
        (1.0 - up.dot(sun_direction)).powf(5.0).clamp(0.0, 1.0),
    );

    // Composition + solar disc
    let sun_angular_diameter_cos = SUN_ANGULAR_DIAMETER_DEGREES.cos();
    let sundisk = smoothstep(
        sun_angular_diameter_cos,
        sun_angular_diameter_cos + 0.00002,
        cos_theta,
    );
    let mut l0 = 0.1 * fex;
    l0 += sun_e * 19000.0 * fex * sundisk;

    lin + l0
}
