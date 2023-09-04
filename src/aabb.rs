use std::ops::BitAnd;

use glam::Vec3A;

use crate::Ray;

#[derive(Default, Clone, Copy, Debug)]
pub struct Aabb {
    pub min: Vec3A,
    pub max: Vec3A,
}

impl Aabb {
    pub fn new(min: Vec3A, max: Vec3A) -> Self {
        Self { min, max }
    }

    pub fn from_point(point: Vec3A) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    pub fn contains_point(&self, point: Vec3A) -> bool {
        (point.cmpge(self.min).bitand(point.cmple(self.max))).all()
    }

    #[inline]
    pub fn extend(&mut self, point: Vec3A) -> &mut Self {
        self.extend_aabb(&Self::from_point(point))
    }

    #[inline]
    pub fn extend_aabb(&mut self, other: &Self) -> &mut Self {
        self.min = self.min.min(other.min);
        self.max = self.max.max(other.max);
        self
    }

    #[inline]
    pub fn clamp_aabb(&mut self, other: &Self) -> &mut Self {
        self.min = self.min.max(other.min);
        self.max = self.max.min(other.max);
        self
    }

    #[inline]
    pub fn diagonal(&self) -> Vec3A {
        self.max - self.min
    }

    #[inline]
    pub fn center(&self) -> Vec3A {
        (self.max + self.min) * 0.5
    }

    #[inline]
    pub fn center_axis(&self, axis: usize) -> f32 {
        (self.max[axis] + self.min[axis]) * 0.5
    }

    #[inline]
    pub fn largest_axis(&self) -> usize {
        let d = self.diagonal();
        if d.x < d.y {
            if d.y < d.z {
                2
            } else {
                1
            }
        } else if d.x < d.z {
            2
        } else {
            0
        }
    }

    pub fn half_area(&self) -> f32 {
        let d = self.diagonal();
        (d.x + d.y) * d.z + d.x * d.y
    }

    pub fn surface_area(&self) -> f32 {
        let d = self.diagonal();
        2.0 * d.dot(d)
    }

    pub fn empty() -> Self {
        Self {
            min: Vec3A::new(f32::MAX, f32::MAX, f32::MAX),
            max: Vec3A::new(f32::MIN, f32::MIN, f32::MIN),
        }
    }

    pub fn intersect(&self, ray: &Ray) -> f32 {
        let t1 = (self.min - ray.origin) * ray.inv_direction;
        let t2 = (self.max - ray.origin) * ray.inv_direction;

        let tmin = t1.min(t2);
        let tmax = t1.max(t2);

        let tmin_n = tmin.x.max(tmin.y.max(tmin.z));
        let tmax_n = tmax.x.min(tmax.y.min(tmax.z));

        if tmax_n >= tmin_n && tmax_n >= 0.0 {
            tmin_n
        } else {
            f32::MAX
        }
    }

    pub fn aabb_intersect(&self, other: &Aabb) -> bool {
        !(self.min.cmpgt(other.max).any() || self.max.cmplt(other.min).any())
    }
}

pub struct AABBIntersection {
    pub tmin: f32,
    pub tmax: f32,
}

impl AABBIntersection {
    #[inline]
    pub fn intersected(&self) -> bool {
        self.tmin <= self.tmax
    }
}
