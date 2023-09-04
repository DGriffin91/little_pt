use glam::{vec2, Vec2, Vec3A};
use std::f32::INFINITY;

use crate::Ray;

#[derive(Clone, Copy, Default)]
pub struct Triangle(pub [Vec3A; 3]);

impl Triangle {
    pub fn compute_normal(&self) -> Vec3A {
        let v1 = self.0[1] - self.0[0];
        let v2 = self.0[2] - self.0[0];
        v1.cross(v2).normalize()
    }

    // https://madmann91.github.io/2021/04/29/an-introduction-to-bvhs.html
    // (Doesn't seem to work well on the GPU, issues with NaN)
    pub fn intersect(&self, ray: &Ray) -> (f32, Vec2) {
        let e1 = self.0[0] - self.0[1];
        let e2 = self.0[2] - self.0[0];
        let n = e1.cross(e2);

        let c = self.0[0] - ray.origin;
        let r = ray.direction.cross(c);
        let inv_det = 1.0 / n.dot(ray.direction);

        let u = r.dot(e2) * inv_det;
        let v = r.dot(e1) * inv_det;
        let w = 1.0 - u - v;

        // These comparisons are designed to return false
        // when one of t, u, or v is a NaN
        if u >= 0.0 && v >= 0.0 && w >= 0.0 {
            let t = n.dot(c) * inv_det;
            if t >= ray.tmin && t <= ray.tmax {
                return (t, vec2(u, v));
            }
        }

        (INFINITY, vec2(u, v))
    }
}
