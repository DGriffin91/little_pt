use std::f32::consts::{PI, TAU};

use glam::{mat3a, vec3a, Mat3A, UVec2, Vec2, Vec3A};

pub fn uhash(a: u32, b: u32) -> u32 {
    let mut x = (a.overflowing_mul(1597334673u32).0) ^ (b.overflowing_mul(3812015801u32).0);
    // from https://nullprogram.com/blog/2018/07/31/
    x = x ^ (x >> 16u32);
    x = x.overflowing_mul(0x7feb352du32).0;
    x = x ^ (x >> 15u32);
    x = x.overflowing_mul(0x846ca68bu32).0;
    x = x ^ (x >> 16u32);
    x
}

pub fn unormf(n: u32) -> f32 {
    n as f32 * (1.0 / 0xffffffffu32 as f32)
}

pub fn hash_noise(ufrag_coord: UVec2, frame: u32) -> f32 {
    let urnd = uhash(ufrag_coord.x, (ufrag_coord.y << 11u32) + frame);
    unormf(urnd)
}

pub fn build_orthonormal_basis(n: Vec3A) -> Mat3A {
    let b1: Vec3A;
    let b2: Vec3A;

    if n.z < 0.0 {
        let a = 1.0 / (1.0 - n.z);
        let b = n.x * n.y * a;
        b1 = vec3a(1.0 - n.x * n.x * a, -b, n.x);
        b2 = vec3a(b, n.y * n.y * a - 1.0, -n.y);
    } else {
        let a = 1.0 / (1.0 + n.z);
        let b = -n.x * n.y * a;
        b1 = vec3a(1.0 - n.x * n.x * a, b, -n.x);
        b2 = vec3a(b, 1.0 - n.y * n.y * a, -n.y);
    }

    mat3a(
        vec3a(b1.x, b1.y, b1.z),
        vec3a(b2.x, b2.y, b2.z),
        vec3a(n.x, n.y, n.z),
    )
}

pub fn cosine_sample_hemisphere(urand: Vec2) -> Vec3A {
    let r = urand.x.sqrt();
    let theta = urand.y * TAU;

    let x = r * theta.cos();
    let y = r * theta.sin();

    vec3a(x, y, 0.0f32.max(1.0 - urand.x).sqrt())
}

pub fn uniform_sample_sphere(urand: Vec2) -> Vec3A {
    let theta = 2.0 * PI * urand.y;
    let z = 1.0 - 2.0 * urand.x;
    let xy = (1.0 - z * z).max(0.0).sqrt();
    let sn = theta.sin();
    let cs = theta.cos();
    vec3a(cs * xy, sn * xy, z)
}

pub fn uniform_sample_disc(urand: Vec2) -> Vec3A {
    let r = urand[0].sqrt();
    let theta = urand[1] * TAU;

    let x = r * theta.cos();
    let y = r * theta.sin();

    Vec3A::new(x, y, 0.0)
}

pub fn uniform_sample_cone(urand: Vec2, cos_theta_max: f32) -> Vec3A {
    let cos_theta = (1.0 - urand.x) + urand.x * cos_theta_max;
    let sin_theta = (1.0 - cos_theta * cos_theta).clamp(0.0, 1.0).sqrt();
    let phi = urand.y * TAU;
    return vec3a(sin_theta * phi.cos(), sin_theta * phi.sin(), cos_theta);
}

pub fn get_f0(reflectance: f32, metallic: f32, base_color: Vec3A) -> Vec3A {
    0.16 * reflectance * reflectance * (1.0 - metallic) + base_color * metallic
}

pub fn perceptual_roughness_to_roughness(perceptual_roughness: f32) -> f32 {
    let clamped_perceptual_roughness = perceptual_roughness.clamp(0.089, 1.0);
    clamped_perceptual_roughness * clamped_perceptual_roughness
}

pub fn reflect(i: Vec3A, n: Vec3A) -> Vec3A {
    i - 2.0 * Vec3A::dot(n, i) * n
}

pub fn smoothstep(e0: f32, e1: f32, x: f32) -> f32 {
    let t = ((x - e0) / (e1 - e0)).max(0.0).min(1.0);
    t * t * (3.0 - 2.0 * t)
}

pub fn smoothstep_vec3(e0: Vec3A, e1: Vec3A, x: Vec3A) -> Vec3A {
    let t = ((x - e0) / (e1 - e0)).clamp(Vec3A::ZERO, Vec3A::ONE);
    t * t * (Vec3A::splat(3.0) - Vec3A::splat(2.0) * t)
}

pub fn powsafe(color: Vec3A, power: f32) -> Vec3A {
    color.abs().powf(power) * color.signum()
}

// Sampling Visible GGX Normals with Spherical Caps
// https://arxiv.org/pdf/2306.05044.pdf

// Helper function: sample the visible hemisphere from a spherical cap
fn sample_vndf_hemisphere(u: Vec2, wi: Vec3A) -> Vec3A {
    // sample a spherical cap in (-wi.z, 1]
    let phi = 2.0 * PI * u.x;
    let z = (1.0 - u.y) * (1.0 + wi.z) - wi.z;
    let sin_theta = (1.0 - z * z).clamp(0.0, 1.0).sqrt();
    let x = sin_theta * phi.cos();
    let y = sin_theta * phi.sin();
    let c = Vec3A::new(x, y, z);
    // compute halfway direction;
    let h = c + wi;
    // return without normalization as this is done later
    h
}

// Sample the GGX VNDF
fn sample_vndf_ggx(urand: Vec2, wi: Vec3A, alpha: Vec2) -> Vec3A {
    let u = Vec2::new(urand.y, urand.x);
    // warp to the hemisphere configuration
    let wi_std = Vec3A::new(wi.x * alpha.x, wi.y * alpha.y, wi.z).normalize();
    // sample the hemisphere
    let wm_std = sample_vndf_hemisphere(u, wi_std);
    // warp back to the ellipsoid configuration
    let wm = Vec3A::new(wm_std.x * alpha.x, wm_std.y * alpha.y, wm_std.z).normalize();
    // return final normal
    wm
}

// ------------------------
// BRDF stuff from kajiya
// ------------------------

#[derive(Debug)]
pub struct NdfSample {
    pub m: Vec3A,
    pub pdf: f32,
}

#[derive(Debug)]
pub struct BrdfSample {
    pub value_over_pdf: Vec3A,
    pub value: Vec3A,
    pub pdf: f32,
    pub transmission_fraction: Vec3A,
    pub wi: Vec3A,
    pub approx_roughness: f32,
}

#[derive(Debug)]
pub struct SmithShadowingMasking {
    pub g: f32,
    pub g_over_g1_wo: f32,
}

fn ggx_ndf(a2: f32, cos_theta: f32) -> f32 {
    let denom_sqrt = cos_theta * cos_theta * (a2 - 1.0) + 1.0;
    a2 / (PI * denom_sqrt * denom_sqrt)
}

fn g_smith_ggx1(ndotv: f32, a2: f32) -> f32 {
    let tan2_v = (1.0 - ndotv * ndotv) / (ndotv * ndotv);
    2.0 / (1.0 + (1.0 + a2 * tan2_v).sqrt())
}

fn pdf_ggx_vn(a2: f32, wo: Vec3A, h: Vec3A) -> f32 {
    let g1 = g_smith_ggx1(wo.z, a2);
    let d = ggx_ndf(a2, h.z);
    g1 * d * (wo.dot(h)).max(0.0) / wo.z
}

// From https://github.com/h3r2tic/kajiya/blob/d3b6ac22c5306cc9d3ea5e2d62fd872bea58d8d6/assets/shaders/inc/brdf.hlsl#LL182C1-L214C6
// https://github.com/NVIDIAGameWorks/Falcor/blob/c0729e806045731d71cfaae9d31a992ac62070e7/Source/Falcor/Experimental/Scene/Material/Microfacet.slang
// https://jcgt.org/published/0007/04/01/paper.pdf
fn sample_vndf(alpha: f32, wo: Vec3A, urand: Vec2) -> NdfSample {
    let h = sample_vndf_ggx(urand, wo, Vec2::new(alpha, alpha));
    let a2 = alpha * alpha;
    let pdf = pdf_ggx_vn(a2, wo, h);

    NdfSample { m: h, pdf }
}

fn eval_fresnel_schlick(f0: Vec3A, f90: Vec3A, cos_theta: f32) -> Vec3A {
    f0.lerp(f90, (1.0 - cos_theta).max(0.0).powi(5))
}

fn smith_shadowing_masking_eval(ndotv: f32, ndotl: f32, a2: f32) -> SmithShadowingMasking {
    let g = g_smith_ggx1(ndotl, a2) * g_smith_ggx1(ndotv, a2);
    let g_over_g1_wo = g_smith_ggx1(ndotl, a2);
    SmithShadowingMasking { g, g_over_g1_wo }
}

const BRDF_SAMPLING_MIN_COS: f32 = 1e-5;

pub fn brdf_sample(roughness: f32, f0: Vec3A, wo: Vec3A, urand: Vec2) -> BrdfSample {
    let ndf_sample = sample_vndf(roughness, wo, urand);
    let wi = reflect(-wo, ndf_sample.m);

    if ndf_sample.m.z <= BRDF_SAMPLING_MIN_COS
        || wi.z <= BRDF_SAMPLING_MIN_COS
        || wo.z <= BRDF_SAMPLING_MIN_COS
    {
        return BrdfSample {
            value_over_pdf: Vec3A::ZERO,
            value: Vec3A::ZERO,
            pdf: 0.0,
            transmission_fraction: Vec3A::ZERO,
            wi: Vec3A::new(0.0, 0.0, -1.0),
            approx_roughness: 0.0,
        };
    }

    let jacobian = 1.0 / (4.0 * wi.dot(ndf_sample.m));
    let fresnel = eval_fresnel_schlick(f0, Vec3A::splat(1.0), ndf_sample.m.dot(wi));
    let a2 = roughness * roughness;
    let cos_theta = ndf_sample.m.z;
    let shadowing_masking = smith_shadowing_masking_eval(wo.z, wi.z, a2);

    let pdf = ndf_sample.pdf * jacobian / wi.z;
    let transmission_fraction = Vec3A::splat(1.0) - fresnel;
    let value = fresnel * shadowing_masking.g * ggx_ndf(a2, cos_theta) / (4.0 * wo.z * wi.z);
    let approx_roughness = roughness;

    BrdfSample {
        value_over_pdf: fresnel * shadowing_masking.g_over_g1_wo,
        value,
        pdf,
        transmission_fraction,
        wi,
        approx_roughness,
    }
}
