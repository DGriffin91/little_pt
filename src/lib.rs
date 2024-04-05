use std::time::Instant;

use glam::{uvec3, vec3a, Vec2, Vec3A};
use image::{ImageBuffer, Rgb};
use obj::{Obj, ObjMaterial};
use renderer::Camera;
use sampling::{get_f0, perceptual_roughness_to_roughness};
use serde::{Deserialize, Serialize};

use crate::{
    aabb::Aabb,
    bvh2::Bvh2,
    renderer::render,
    svenstaro::{svenstaro_bbox_shapes, traverse_svenstaro},
    triangle::Triangle,
};

pub mod aabb;
pub mod bvh2;
pub mod d3_image;
pub mod renderer;
pub mod sampling;
pub mod sky;
pub mod svenstaro;
pub mod tonemapping;
pub mod triangle;

pub fn safe_inverse(x: f32) -> f32 {
    if x.abs() <= f32::EPSILON {
        x.signum() / f32::EPSILON
    } else {
        1.0 / x
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Ray {
    pub origin: Vec3A,
    pub direction: Vec3A,
    pub inv_direction: Vec3A,
    pub tmin: f32,
    pub tmax: f32,
}

impl Ray {
    pub fn new(origin: Vec3A, direction: Vec3A, min: f32, max: f32) -> Self {
        Ray {
            origin,
            direction,
            inv_direction: vec3a(
                safe_inverse(direction.x),
                safe_inverse(direction.y),
                safe_inverse(direction.z),
            ),
            tmin: min,
            tmax: max,
        }
    }
}

#[derive(Clone, Copy)]
pub struct Hit {
    pub material_index: u32,
    pub prim_index: u32,
    pub distance: f32,
    pub uv: Vec2,
}

impl Hit {
    pub fn none() -> Self {
        Self {
            material_index: 0,
            prim_index: u32::MAX,
            distance: f32::MAX,
            uv: Vec2::ZERO,
        }
    }
}

#[derive(Debug)]
pub struct Material {
    pub base_color: Vec3A,
    pub perceptual_roughness: f32,
    pub roughness: f32,
    pub f0: Vec3A,
    pub metallic: f32,
}

impl Material {
    pub fn from_mtl(mtl: ObjMaterial) -> Self {
        match mtl {
            ObjMaterial::Ref(_) => Default::default(),
            ObjMaterial::Mtl(mtl) => {
                let illum = mtl.illum.unwrap_or(2);
                let metallic = if (3..=8).contains(&illum) { 1.0 } else { 0.0 };
                let clamped_ns = mtl.ns.unwrap_or(250.0).clamp(0.0, 1000.0);
                let perceptual_roughness = 1.0 - (clamped_ns / 1000.0).sqrt();
                let base_color = Vec3A::from(mtl.kd.unwrap_or([0.5; 3])).max(Vec3A::ZERO);
                Material {
                    base_color,
                    perceptual_roughness,
                    metallic,
                    roughness: perceptual_roughness_to_roughness(perceptual_roughness),
                    f0: get_f0(0.5, metallic, base_color),
                }
            }
        }
    }
}

impl Default for Material {
    fn default() -> Self {
        Material {
            base_color: Vec3A::splat(0.5),
            perceptual_roughness: 0.5,
            metallic: 0.0,
            roughness: perceptual_roughness_to_roughness(0.5),
            f0: get_f0(0.5, 0.0, Vec3A::splat(0.5)),
        }
    }
}

#[derive(Debug, Deserialize, Serialize)]
pub struct Scene {
    pub model_path: String,
    pub camera: Camera,
    pub sun_direction: Vec3A,
}

impl Scene {
    pub fn render(
        &self,
        width: u32,
        height: u32,
        samples: u32,
        trace_recursion: u32,
    ) -> ImageBuffer<Rgb<u8>, Vec<u8>> {
        let start_time = Instant::now();
        let width = width as f32;
        let height = height as f32;

        let mut objf = Obj::load(&self.model_path).unwrap();
        let _ = objf.load_mtls();
        println!("Import {}: {:?}", self.model_path, start_time.elapsed());

        let start_time = Instant::now();
        let mut aabbs = Vec::new();
        let mut centers = Vec::new();
        let mut tris = Vec::new();
        let mut material_indices = Vec::new();
        let mut materials = vec![Material::default()];

        for obj in objf.data.objects {
            for group in obj.groups {
                let mut current_mat = 0;
                if let Some(mtl) = group.material {
                    current_mat = materials.len() as u32;
                    materials.push(Material::from_mtl(mtl));
                }
                for poly in group.polys {
                    let a = objf.data.position[poly.0[0].0].into();
                    let b = objf.data.position[poly.0[1].0].into();
                    let c = objf.data.position[poly.0[2].0].into();
                    let aabb = *Aabb::empty().extend(a).extend(b).extend(c);
                    let center = (a + b + c) / 3.0;
                    let tri = Triangle([a, b, c]);
                    aabbs.push(aabb);
                    centers.push(center);
                    tris.push(tri);
                    material_indices.push(current_mat);
                }
            }
        }
        println!("Loading: {:?}", start_time.elapsed());

        println!(
            "{}: {}x{} {} samples {} tris",
            self.model_path,
            width,
            height,
            samples,
            tris.len()
        );

        let mut imgbuf = image::ImageBuffer::new(width as u32, height as u32);

        let mut shapes = svenstaro_bbox_shapes(&tris);
        let build_time = Instant::now();
        //let bvh2 = Bvh2::build(&aabbs, &centers);
        let bvh2 = bvh::bvh::Bvh::build(&mut shapes);
        println!("Build BVH: {:?}", build_time.elapsed());

        let trace_time = Instant::now();
        render(
            &mut imgbuf,
            width,
            height,
            samples,
            self,
            &materials,
            |ray: &Ray| {
                //let mut hit = bvh2.traverse(ray, &tris);
                let mut hit = traverse_svenstaro(&bvh2, &shapes, ray);
                let prim_index = hit.prim_index as usize;
                let tri = if hit.distance < f32::MAX {
                    hit.material_index = material_indices[prim_index];
                    tris[prim_index]
                } else {
                    Triangle::default()
                };
                (hit, tri)
            },
            trace_recursion,
        );
        let tlas_threaded_voxel_blas_bvh_trace_time = trace_time.elapsed().as_secs_f32();
        println!("Trace: {:?}", tlas_threaded_voxel_blas_bvh_trace_time);
        imgbuf
    }
}
