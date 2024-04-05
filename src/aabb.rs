use std::ops::BitAnd;

use glam::Vec3A;

use crate::Ray;

#[derive(Default, Clone, Copy, Debug, PartialEq)]
pub struct Aabb {
    pub min: Vec3A,
    pub max: Vec3A,
}

impl Aabb {
    /// Empty AABB
    pub const INVALID: Self = Self {
        min: Vec3A::splat(f32::MAX),
        max: Vec3A::splat(f32::MIN),
    };

    pub const LARGEST: Self = Self {
        min: Vec3A::splat(-f32::MAX),
        max: Vec3A::splat(f32::MAX),
    };

    pub const INFINITY: Self = Self {
        min: Vec3A::splat(-f32::INFINITY),
        max: Vec3A::splat(f32::INFINITY),
    };

    pub fn new(min: Vec3A, max: Vec3A) -> Self {
        Self { min, max }
    }

    pub fn from_point(point: Vec3A) -> Self {
        Self {
            min: point,
            max: point,
        }
    }

    pub fn from_points(points: &[Vec3A]) -> Self {
        let mut points = points.iter();
        let mut aabb = Aabb::from_point(*points.next().unwrap());
        for point in points {
            aabb.extend(*point);
        }
        aabb
    }

    pub fn contains_point(&self, point: Vec3A) -> bool {
        (point.cmpge(self.min).bitand(point.cmple(self.max))).all()
    }

    #[inline]
    pub fn extend(&mut self, point: Vec3A) -> &mut Self {
        *self = self.union(&Self::from_point(point));
        self
    }

    #[inline]
    pub fn union(&self, other: &Self) -> Self {
        Aabb {
            min: self.min.min(other.min),
            max: self.max.max(other.max),
        }
    }

    #[inline]
    pub fn intersection(&self, other: &Self) -> Self {
        Aabb {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        }
    }

    #[inline]
    pub fn clamp_aabb(&self, other: &Self) -> Self {
        Aabb {
            min: self.min.max(other.min),
            max: self.max.min(other.max),
        }
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

    #[inline]
    pub fn smallest_axis(&self) -> usize {
        let d = self.diagonal();
        if d.x > d.y {
            if d.y > d.z {
                2
            } else {
                1
            }
        } else if d.x > d.z {
            2
        } else {
            0
        }
    }

    #[inline]
    pub fn half_area(&self) -> f32 {
        let d = self.diagonal();
        (d.x + d.y) * d.z + d.x * d.y
    }

    #[inline]
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

    pub fn aabb_intersect(&self, other: &Aabb) -> bool {
        !(self.min.cmpgt(other.max).any() || self.max.cmplt(other.min).any())
    }

    pub fn ray_intersect(&self, ray: &Ray) -> f32 {
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
