use core::f32::consts::PI;
use glam::{vec2, Vec2, Vec3A};

fn rsi(rd: Vec3A, r0: Vec3A, sr: f32) -> Vec2 {
    // ray-sphere intersection that assumes
    // the sphere is centered at the origin.
    // No intersection when result.x > result.y
    let a = rd.dot(rd);
    let b = 2.0 * rd.dot(r0);
    let c = r0.dot(r0) - (sr * sr);
    let d = (b * b) - (4.0 * a * c);

    if d < 0.0 {
        return vec2(1e5, -1e5);
    } else {
        return vec2((-b - d.sqrt()) / (2.0 * a), (-b + d.sqrt()) / (2.0 * a));
    }
}

pub struct Nishita {
    /// Ray Origin (Default: `(0.0, 6372e3, 0.0)`).
    ///
    /// Controls orientation of the sky and height of the sun.
    /// It can be thought of as the up-axis and values should be somewhere between planet radius and atmosphere radius (with a bias towards lower values).
    /// When used with `planet_radius` and `atmosphere_radius`, it can be used to change sky brightness and falloff
    pub ray_origin: Vec3A,

    /// Sun Intensity (Default: `22.0`).
    ///
    /// Controls how intense the sun's brightness is.
    pub sun_intensity: f32,

    /// Planet Radius (Default: `6371e3`).
    ///
    /// Controls the radius of the planet.
    /// Heavily interdependent with `atmosphere_radius`
    pub planet_radius: f32,

    /// Atmosphere Radius (Default: `6471e3`).
    ///
    /// Controls the radius of the atmosphere.
    /// Heavily interdependent with `planet_radius`.
    pub atmosphere_radius: f32,

    /// Rayleigh Scattering Coefficient (Default: `(5.5e-6, 13.0e-6, 22.4e-6)`).
    ///
    /// Strongly influences the color of the sky.
    pub rayleigh_coefficient: Vec3A,

    /// Rayleigh Scattering Scale Height (Default: `8e3`).
    ///
    /// Controls the amount of Rayleigh scattering.
    pub rayleigh_scale_height: f32,

    /// Mie Scattering Coefficient (Default: `21e-6`).
    ///
    /// Strongly influences the color of the horizon.
    pub mie_coefficient: f32,

    /// Mie Scattering Scale Height (Default: `1.2e3`).
    ///
    /// Controls the amount of Mie scattering.
    pub mie_scale_height: f32,

    /// Mie Scattering Preferred Direction (Default: `0.758`).
    ///
    /// Controls the general direction of Mie scattering.
    pub mie_direction: f32,
}

const ISTEPS: u32 = 16;
const JSTEPS: u32 = 8;

impl Nishita {
    pub fn render(&self, dir: Vec3A, sun_position: Vec3A) -> Vec3A {
        // Normalize the ray direction and sun position.
        let r = dir.normalize();
        let p_sun = sun_position.normalize();

        // Calculate the step size of the primary ray.
        let mut p = rsi(r, self.ray_origin, self.atmosphere_radius);
        if p.x > p.y {
            return Vec3A::splat(0.0);
        }
        p.y = p.y.min(rsi(r, self.ray_origin, self.planet_radius).x);
        let i_step_size = (p.y - p.x) / (ISTEPS as f32);

        // Initialize the primary ray depth.
        let mut i_depth = 0.0;

        // Initialize accumulators for Rayleigh and Mie scattering.
        let mut total_rlh = Vec3A::splat(0.0);
        let mut total_mie = Vec3A::splat(0.0);

        // Initialize optical depth accumulators for the primary ray.
        let mut i_od_rlh = 0.0;
        let mut i_od_mie = 0.0;

        // Calculate the Rayleigh and Mie phases.
        let mu = r.dot(p_sun);
        let mumu = mu * mu;
        let gg = self.mie_direction * self.mie_direction;
        let p_rlh = 3.0 / (16.0 * PI) * (1.0 + mumu);
        let p_mie = 3.0 / (8.0 * PI) * ((1.0 - gg) * (mumu + 1.0))
            / ((1.0 + gg - 2.0 * mu * self.mie_direction).powf(1.5) * (2.0 + gg));

        // Sample the primary ray.
        for _ in 0..ISTEPS {
            // Calculate the primary ray sample position.
            let i_pos = self.ray_origin + r * (i_depth + i_step_size * 0.5);

            // Calculate the height of the sample.
            let i_height = i_pos.length() - self.planet_radius;

            // Calculate the optical depth of the Rayleigh and Mie scattering for this step.
            let od_step_rlh = (-i_height / self.rayleigh_scale_height).exp() * i_step_size;
            let od_step_mie = (-i_height / self.mie_scale_height).exp() * i_step_size;

            // Accumulate optical depth.
            i_od_rlh += od_step_rlh;
            i_od_mie += od_step_mie;

            // Calculate the step size of the secondary ray.
            let j_step_size = rsi(p_sun, i_pos, self.atmosphere_radius).y / (JSTEPS as f32);

            // Initialize the secondary ray depth.
            let mut j_depth = 0.0;

            // Initialize optical depth accumulators for the secondary ray.
            let mut j_od_rlh = 0.0;
            let mut j_od_mie = 0.0;

            // Sample the secondary ray.
            for _ in 0..JSTEPS {
                // Calculate the secondary ray sample position.
                let j_pos = i_pos + p_sun * (j_depth + j_step_size * 0.5);

                // Calculate the height of the sample.
                let j_height = j_pos.length() - self.planet_radius;

                // Accumulate the optical depth.
                j_od_rlh += (-j_height / self.rayleigh_scale_height).exp() * j_step_size;
                j_od_mie += (-j_height / self.mie_scale_height).exp() * j_step_size;

                // Increment the secondary ray depth.
                j_depth += j_step_size;
            }

            // Calculate attenuation.
            let attn = (-(self.mie_coefficient * (i_od_mie + j_od_mie)
                + self.rayleigh_coefficient * (i_od_rlh + j_od_rlh)))
                .exp();

            // Accumulate scattering.
            total_rlh += od_step_rlh * attn;
            total_mie += od_step_mie * attn;

            // Increment the primary ray depth.
            i_depth += i_step_size;
        }

        // Calculate and return the final color.
        return self.sun_intensity
            * (p_rlh * self.rayleigh_coefficient * total_rlh
                + p_mie * self.mie_coefficient * total_mie);
    }
}

impl Default for Nishita {
    fn default() -> Self {
        Self {
            ray_origin: Vec3A::new(0.0, 6372e3, 0.0),
            sun_intensity: 150.0,
            planet_radius: 6371e3,
            atmosphere_radius: 6471e3,
            rayleigh_coefficient: Vec3A::new(5.5e-6, 13.0e-6, 22.4e-6),
            rayleigh_scale_height: 8e3,
            mie_coefficient: 21e-6,
            mie_scale_height: 1.2e3,
            mie_direction: 0.758,
        }
    }
}
