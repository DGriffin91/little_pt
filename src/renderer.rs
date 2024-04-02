use glam::{uvec2, vec2, vec3a, vec4, UVec2, Vec3A, Vec4Swizzles};
use image::{ImageBuffer, Rgb};
use rayon::prelude::{IntoParallelRefMutIterator, ParallelIterator};
use serde::{Deserialize, Serialize};

use crate::{
    sampling::{
        brdf_sample, build_orthonormal_basis, cosine_sample_hemisphere, hash_noise,
        uniform_sample_cone, BrdfSample,
    },
    sky::{Sky, SUN_ANGULAR_DIAMETER},
    tonemapping::tony_mc_mapface,
    triangle::Triangle,
    Hit, Material, Ray, Scene,
};

pub const EPSILON: f32 = 0.00001;

#[derive(Debug, Deserialize, Serialize)]
pub struct Camera {
    pub eye: Vec3A,
    pub dir: Vec3A,
    pub fov: f32,
    pub exposure: f32,
}

impl Camera {
    fn get_ray(&self, x: f32, y: f32, width: f32, height: f32) -> Ray {
        let up = Vec3A::Y;
        let right = self.dir.cross(up).normalize();

        let aspect_ratio = width / height;
        let fov_adj = (self.fov.to_radians() / 2.0).tan();
        let u = (x / width * 2.0 - 1.0) * fov_adj * aspect_ratio;
        let v = ((1.0 - y / height) * 2.0 - 1.0) * fov_adj;
        let direction = (self.dir + u * right + v * up).normalize();
        Ray::new(self.eye, direction, 0.0, f32::MAX)
    }
}

fn img_col_vec3(v: Vec3A) -> image::Rgb<u8> {
    image::Rgb([
        (v.x * 255.0) as u8,
        (v.y * 255.0) as u8,
        (v.z * 255.0) as u8,
    ])
}

const NORMAL_BIAS: f32 = 0.0001;
const DEPTH_BIAS: f32 = 0.99999;

pub fn render<F>(
    imgbuf: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    width: f32,
    height: f32,
    samples: u32,
    scene: &Scene,
    materials: &[Material],
    traverse_fn: F,
    trace_recursion: u32,
) where
    F: Fn(&Ray) -> (Hit, Triangle) + Send + Sync,
{
    let mut xy_pixel_vec: Vec<(u32, u32, &mut Rgb<u8>)> = imgbuf.enumerate_pixels_mut().collect();

    xy_pixel_vec
        .par_iter_mut()
        .for_each(|(center_x, center_y, pixel)| {
            let mut col = Vec3A::ZERO;

            for i in 0..samples {
                let xy_offset = vec2(
                    hash_noise(UVec2::ZERO, i + 70000),
                    hash_noise(UVec2::ZERO, i + 80000),
                ) - 0.5;
                let (x, y) = (
                    *center_x as f32 + xy_offset.x,
                    *center_y as f32 + xy_offset.y,
                );
                let ufragcoord = uvec2(*center_x, *center_y);

                let ray = scene.camera.get_ray(x, y, width, height);

                col += render_ray(
                    ufragcoord,
                    i,
                    scene,
                    &traverse_fn,
                    ray,
                    materials,
                    trace_recursion,
                );
            }
            col /= samples as f32;

            col *= Vec3A::splat(2.0).powf(scene.camera.exposure);
            //let col = col / (col + Vec3A::ONE); // reinhard
            //col = somewhat_boring_display_transform(col);
            col = tony_mc_mapface(col);
            col = col.powf(1.0 / 2.2); //Convert to SRGB
            **pixel = img_col_vec3(col);
        });
}

fn render_ray<F>(
    ufragcoord: UVec2,
    sample_n: u32,
    scene: &Scene,
    traverse_fn: &F,
    ray: Ray,
    materials: &[Material],
    recursion_depth: u32,
) -> Vec3A
where
    F: Fn(&Ray) -> (Hit, Triangle) + Send + Sync,
{
    let seed = (recursion_depth + 1) * (sample_n + 1);
    let mut col = Vec3A::ZERO;
    let sun_color = vec3a(1.0, 0.73, 0.46) * 1000000.0;
    //let sky_color = (vec3a(0.875, 0.95, 0.995) * 2.0).powf(2.2);
    let init_sun_dir = scene.sun_direction.normalize_or_zero();

    let nee = 1.0 - SUN_ANGULAR_DIAMETER.cos();
    let sun_rnd = vec2(
        hash_noise(ufragcoord, seed + 10000),
        hash_noise(ufragcoord, seed + 20000),
    );
    let sun_basis = build_orthonormal_basis(init_sun_dir);
    let sun_dir =
        (sun_basis * uniform_sample_cone(sun_rnd, (SUN_ANGULAR_DIAMETER * 0.5).cos())).normalize();

    let (hit, tri) = traverse_fn(&ray);

    if hit.distance < f32::MAX {
        let primary_mat = &materials[hit.material_index as usize];
        let surface_normal = tri.compute_normal();
        let tangent_to_world = build_orthonormal_basis(surface_normal);
        let tangent_to_world_transpose = tangent_to_world.transpose();
        let primary_hitp =
            ray.origin + ray.direction * hit.distance * DEPTH_BIAS + surface_normal * NORMAL_BIAS;
        let v = (ray.origin - primary_hitp).normalize();

        let hit_sun = traverse_fn(&Ray::new(primary_hitp, -sun_dir, 0.0, f32::MAX)).0;
        if hit_sun.distance >= f32::MAX {
            col += (sun_color
                * nee
                * primary_mat.base_color
                * surface_normal.dot(-sun_dir).max(0.00001))
            .max(Vec3A::ZERO);
        }

        if recursion_depth > 0 {
            let urand = vec4(
                hash_noise(ufragcoord, seed + 30000),
                hash_noise(ufragcoord, seed + 40000),
                hash_noise(ufragcoord, seed + 50000),
                hash_noise(ufragcoord, seed + 60000),
            );

            // Diffuse
            let mut diffuse_dir = cosine_sample_hemisphere(urand.xy());
            diffuse_dir = (tangent_to_world * diffuse_dir).normalize();

            let diffuse_ray = Ray::new(primary_hitp, diffuse_dir, 0.0, f32::MAX);
            let diffuse_hit_color = render_ray(
                ufragcoord,
                sample_n,
                scene,
                traverse_fn,
                diffuse_ray,
                materials,
                recursion_depth - 1,
            );
            col += (diffuse_hit_color * primary_mat.base_color).max(Vec3A::ZERO);

            // Specular
            let wo = v;
            let mut brdf_s = BrdfSample::invalid();
            // VNDF still returns a lot of invalid samples on rough surfaces at F0 angles!
            // https://github.com/EmbarkStudios/kajiya/blob/d373f76b8a2bff2023c8f92b911731f8eb49c6a9/assets/shaders/rtr/reflection.rgen.hlsl#L107
            for _ in 0..4 {
                brdf_s = brdf_sample(
                    primary_mat.roughness,
                    primary_mat.f0,
                    tangent_to_world_transpose * wo,
                    urand.zw(),
                );
                if brdf_s.wi.z > 0.000001 {
                    break;
                }
            }
            if !brdf_s.wi.is_finite() {
                return col;
            }

            let spec_dir = tangent_to_world * brdf_s.wi;
            let spec_ray = Ray::new(primary_hitp, spec_dir, 0.0, f32::MAX);
            let spec_hit_color = render_ray(
                ufragcoord,
                sample_n,
                scene,
                traverse_fn,
                spec_ray,
                materials,
                recursion_depth - 1,
            );
            col += (spec_hit_color * brdf_s.value_over_pdf).max(Vec3A::ZERO);
        }
    } else {
        //col += sky_color;
        // Sun results in fireflies. Clamp to avoid randomly sampling super high values.
        // TODO, don't want to do this when sampling 1st bounce specular.
        col += Sky::red_sunset2()
            .render(ray.direction, -init_sun_dir)
            .clamp(Vec3A::ZERO, Vec3A::splat(100.0));
    }

    col
}

pub fn render_normals<F>(
    imgbuf: &mut ImageBuffer<Rgb<u8>, Vec<u8>>,
    width: f32,
    height: f32,
    _samples: u32,
    cam: &Camera,
    _sun_dir: &Vec3A,
    traverse_fn: F,
) where
    F: Fn(&Ray) -> (Hit, Triangle) + Send + Sync,
{
    let mut xy_pixel_vec: Vec<(u32, u32, &mut Rgb<u8>)> = imgbuf.enumerate_pixels_mut().collect();

    xy_pixel_vec
        .par_iter_mut()
        .for_each(|(center_x, center_y, pixel)| {
            let mut col = Vec3A::ZERO;

            let (x, y) = (*center_x as f32, *center_y as f32);

            let ray = cam.get_ray(x, y, width, height);

            let (hit, tri) = traverse_fn(&ray);

            if hit.distance < f32::MAX {
                col = tri.compute_normal();
            }

            **pixel = img_col_vec3(col);
        });
}
