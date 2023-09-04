use std::f32::consts::PI;

use glam::{vec3a, Vec3A};

use crate::sampling::smoothstep;

pub struct AtmosphereParameters {
    pub depolarization_factor: f32,
    pub mie_coefficient: f32,
    pub mie_directional_g: f32,
    pub mie_k_coefficient: Vec3A,
    pub mie_v: f32,
    pub mie_zenith_length: f32,
    pub num_molecules: f32,
    pub primaries: Vec3A,
    pub rayleigh: f32,
    pub rayleigh_zenith_length: f32,
    pub refractive_index: f32,
    pub sun_angular_diameter_degrees: f32,
    pub sun_intensity_factor: f32,
    pub sun_intensity_falloff_steepness: f32,
    pub turbidity: f32,
}

const A: f32 = 0.15; // Shoulder strength
const B: f32 = 0.50; // Linear strength
const C: f32 = 0.10; // Linear angle
const D: f32 = 0.20; // Toe strength
const E: f32 = 0.02; // Toe numerator
const F: f32 = 0.30; // Toe denominator
pub fn tonemap(w: Vec3A) -> Vec3A {
    return ((w * (A * w + C * B) + D * E) / (w * (A * w + B) + D * F)) - E / F;
}

impl AtmosphereParameters {
    pub fn red_sunset() -> AtmosphereParameters {
        AtmosphereParameters {
            depolarization_factor: 0.02,
            mie_coefficient: 0.005,
            mie_directional_g: 0.82,
            mie_k_coefficient: vec3a(0.686, 0.678, 0.666),
            mie_v: 3.936,
            mie_zenith_length: 34000.0,
            num_molecules: 2.542e25,
            primaries: vec3a(6.8e-7f32, 5.5e-7f32, 4.5e-7f32),
            rayleigh: 2.28,
            rayleigh_zenith_length: 8400.0,
            refractive_index: 1.00029,
            sun_angular_diameter_degrees: 0.00933,
            sun_intensity_factor: 1000.0,
            sun_intensity_falloff_steepness: 1.5,
            turbidity: 4.7,
        }
    }

    pub fn red_sunset2() -> AtmosphereParameters {
        AtmosphereParameters {
            depolarization_factor: 0.02,
            mie_coefficient: 0.008,
            mie_directional_g: 0.62,
            mie_k_coefficient: vec3a(0.686, 0.67, 0.4),
            mie_v: 3.936,
            mie_zenith_length: 34000.0,
            num_molecules: 3.542e25,
            primaries: vec3a(6.8e-7f32, 5.5e-7f32, 4.5e-7f32),
            rayleigh: 2.28,
            rayleigh_zenith_length: 8400.0,
            refractive_index: 1.00029,
            sun_angular_diameter_degrees: 0.00933,
            sun_intensity_factor: 1000.0,
            sun_intensity_falloff_steepness: 1.5,
            turbidity: 4.7,
        }
    }

    pub fn render(&self, dir: Vec3A, sun_position: Vec3A) -> Vec3A {
        let sunfade = 1.0 - (1.0 - (sun_position.y / 450000.0).exp()).clamp(0.0, 1.0);
        let rayleigh_coefficient = self.rayleigh - (1.0 * (1.0 - sunfade));
        let beta_r = self.total_rayleigh(self.primaries) * rayleigh_coefficient;

        let beta_m = self.total_mie(self.primaries) * self.mie_coefficient;

        let zenith_angle = (0.0f32.max(Vec3A::Y.dot(dir))).acos();
        let denom =
            zenith_angle.cos() + 0.15 * (93.885 - ((zenith_angle * 180.0) / PI)).powf(-1.253);
        let s_r = self.rayleigh_zenith_length / denom;
        let s_m = self.mie_zenith_length / denom;

        let fex = (-(beta_r * s_r + beta_m * s_m)).exp();

        let sun_direction = sun_position.normalize();
        let cos_theta = dir.dot(sun_direction);
        let beta_r_theta = beta_r * Self::rayleigh_phase(cos_theta * 0.5 + 0.5);
        let beta_m_theta =
            beta_m * Self::henyey_greenstein_phase(cos_theta, self.mie_directional_g);

        let sun_e = self.sun_intensity(sun_direction.dot(Vec3A::Y));
        let mut lin =
            (sun_e * ((beta_r_theta + beta_m_theta) / (beta_r + beta_m)) * (1.0 - fex)).powf(1.5);
        lin *= Vec3A::splat(1.0).lerp(
            (sun_e * ((beta_r_theta + beta_m_theta) / (beta_r + beta_m)) * fex).powf(0.5),
            (1.0 - Vec3A::Y.dot(sun_direction))
                .powf(5.0)
                .clamp(0.0, 1.0),
        );

        let sun_angular_diameter_cos = (self.sun_angular_diameter_degrees).cos();
        let sundisk = smoothstep(
            sun_angular_diameter_cos,
            sun_angular_diameter_cos, // + 0.00002
            cos_theta,
        );
        let mut l0 = Vec3A::splat(0.1) * fex;
        l0 += sun_e * 19000.0 * fex * sundisk;
        let mut color = (lin + l0) * 0.04;
        let low_falloff = (Vec3A::Y.dot(dir) + 0.4).powf(5.0).max(0.0);
        color = (color * 0.1).powf(3.0) * low_falloff;
        color.powf(1.0 / (1.2 + (1.2 * sunfade))) * 0.5
    }

    fn total_rayleigh(&self, lambda: Vec3A) -> Vec3A {
        (8.0 * PI.powi(3)
            * (self.refractive_index.powi(2) - 1.0).powi(2)
            * (6.0 + 3.0 * self.depolarization_factor))
            / (3.0
                * self.num_molecules
                * lambda.powf(4.0)
                * (6.0 - 7.0 * self.depolarization_factor))
    }

    fn total_mie(&self, lambda: Vec3A) -> Vec3A {
        let c = 0.2 * self.turbidity * 10e-18;
        0.434 * c * PI * (2.0 * PI / lambda).powf(self.mie_v - 2.0) * self.mie_k_coefficient
    }

    fn rayleigh_phase(cos_theta: f32) -> f32 {
        (3.0 / (16.0 * PI)) * (1.0 + cos_theta.powi(2))
    }

    fn henyey_greenstein_phase(cos_theta: f32, g: f32) -> f32 {
        (1.0 / (4.0 * PI)) * ((1.0 - g.powi(2)) / (1.0 - 2.0 * g * cos_theta + g.powi(2)).powf(1.5))
    }

    fn sun_intensity(&self, zenith_angle_cos: f32) -> f32 {
        let cutoff_angle = PI / 1.95;
        self.sun_intensity_factor
            * 0.0f32.max(
                1.0 - (-((cutoff_angle - zenith_angle_cos.acos()).exp()
                    / self.sun_intensity_falloff_steepness)),
            )
    }
}
