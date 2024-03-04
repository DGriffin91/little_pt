use bvh::{bounding_hierarchy::BHShape, ray::Intersection};

use crate::{triangle::Triangle, Hit, Ray};
use bvh::bvh::Bvh;
use glam::vec2;
use nalgebra::{Point, SVector};

pub struct TriShape {
    tri: [Point<f32, 3>; 3],
    shape_index: usize,
    node_index: usize,
}

pub const EPSILON: f32 = 1e-5;

impl bvh::aabb::Bounded<f32, 3> for TriShape {
    fn aabb(&self) -> bvh::aabb::Aabb<f32, 3> {
        let mut aabb = bvh::aabb::Aabb::empty()
            .grow(&self.tri[0])
            .grow(&self.tri[1])
            .grow(&self.tri[2]);
        let size = aabb.size();
        // If the triangle is axis aligned the resulting AABB will have no size on that axis.
        if size.x < EPSILON {
            aabb.max.x += EPSILON;
        }
        if size.y < EPSILON {
            aabb.max.y += EPSILON;
        }
        if size.z < EPSILON {
            aabb.max.z += EPSILON;
        }
        aabb
    }
}

impl BHShape<f32, 3> for TriShape {
    fn set_bh_node_index(&mut self, index: usize) {
        self.node_index = index;
    }

    fn bh_node_index(&self) -> usize {
        self.node_index
    }
}

pub fn traverse_svenstaro(bvh: &Bvh<f32, 3>, shapes: &Vec<TriShape>, ray: &Ray) -> Hit {
    let ray_s = bvh::ray::Ray::new(
        Point::<f32, 3>::from(Into::<[f32; 3]>::into(ray.origin)),
        SVector::<f32, 3>::from(Into::<[f32; 3]>::into(ray.direction)),
    );
    let mut min_dist = f32::MAX;
    let mut closest_hit = Hit::none();
    for hit in bvh.traverse_iterator(&ray_s, shapes) {
        let intersection = intersects_triangle(&ray_s, &hit.tri[0], &hit.tri[1], &hit.tri[2]);
        let hit_dist = intersection.distance;
        let uv = vec2(intersection.u, intersection.v);
        if hit_dist < min_dist {
            min_dist = hit_dist;
            closest_hit = Hit {
                prim_index: hit.shape_index as u32,
                distance: hit_dist,
                uv,
                ..Hit::none()
            };
        }
    }
    closest_hit
}

pub fn svenstaro_bbox_shapes(tris: &[Triangle]) -> Vec<TriShape> {
    let shapes: Vec<TriShape> = tris
        .iter()
        .enumerate()
        .map(|(i, tri)| TriShape {
            tri: [
                Point::<f32, 3>::from(Into::<[f32; 3]>::into(tri.0[0])),
                Point::<f32, 3>::from(Into::<[f32; 3]>::into(tri.0[1])),
                Point::<f32, 3>::from(Into::<[f32; 3]>::into(tri.0[2])),
            ],
            shape_index: i,
            node_index: 0,
        })
        .collect();
    shapes
}

// From https://github.com/svenstaro/bvh/blob/v0.8.0/src/ray/ray_impl.rs#L116
// Copied for version without backface culling
pub fn intersects_triangle(
    ray: &bvh::ray::Ray<f32, 3>,
    a: &Point<f32, 3>,
    b: &Point<f32, 3>,
    c: &Point<f32, 3>,
) -> Intersection<f32> {
    let a_to_b = *b - *a;
    let a_to_c = *c - *a;

    // Begin calculating determinant - also used to calculate u parameter
    // u_vec lies in view plane
    // length of a_to_c in view_plane = |u_vec| = |a_to_c|*sin(a_to_c, dir)
    let u_vec = ray.direction.cross(&a_to_c);

    // If determinant is near zero, ray lies in plane of triangle
    // The determinant corresponds to the parallelepiped volume:
    // det = 0 => [dir, a_to_b, a_to_c] not linearly independant
    let det = a_to_b.dot(&u_vec);

    // Only testing positive bound, thus enabling backface culling
    // If backface culling is not desired write:
    // det < EPSILON && det > -EPSILON
    if det < EPSILON && det > -EPSILON {
        return Intersection::new(f32::INFINITY, 0.0, 0.0);
    }

    let inv_det = 1.0 / det;

    // Vector from point a to ray origin
    let a_to_origin = ray.origin - *a;

    // Calculate u parameter
    let u = a_to_origin.dot(&u_vec) * inv_det;

    // Test bounds: u < 0 || u > 1 => outside of triangle
    if !(0.0..=1.0).contains(&u) {
        return Intersection::new(f32::INFINITY, u, 0.0);
    }

    // Prepare to test v parameter
    let v_vec = a_to_origin.cross(&a_to_b);

    // Calculate v parameter and test bound
    let v = ray.direction.dot(&v_vec) * inv_det;
    // The intersection lies outside of the triangle
    if v < 0.0 || u + v > 1.0 {
        return Intersection::new(f32::INFINITY, u, v);
    }

    let dist = a_to_c.dot(&v_vec) * inv_det;

    if dist > f32::EPSILON {
        Intersection::new(dist, u, v)
    } else {
        Intersection::new(f32::INFINITY, u, v)
    }
}
