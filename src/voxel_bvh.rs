use glam::{IVec3, UVec3, Vec3A};
use rayon::prelude::{IndexedParallelIterator, IntoParallelRefMutIterator, ParallelIterator};

use crate::{aabb::Aabb, bvh2::Bvh2, triangle::Triangle, Hit, Ray};

pub struct VoxelGrid {
    pub voxels: Vec<Bvh2>,
    pub aabb: Aabb,
    pub aabb_size: Vec3A,
    pub voxel_size: Vec3A,
    pub resolution: UVec3,
}

impl VoxelGrid {
    pub fn build(aabbs: &[Aabb], centers: &[Vec3A], resolution: UVec3) -> VoxelGrid {
        let mut gridaabb = Aabb::empty();
        for aabb in aabbs.iter() {
            gridaabb.extend_aabb(aabb);
        }

        let aabb_size = gridaabb.diagonal();
        let mut grid = VoxelGrid {
            aabb_size,
            voxel_size: aabb_size / resolution.as_vec3a(),
            aabb: gridaabb,
            voxels: Vec::new(),
            resolution,
        };

        let mut voxels =
            vec![Bvh2::default(); (resolution.x * resolution.y * resolution.z) as usize];

        voxels
            .par_iter_mut()
            .enumerate()
            .for_each(|(voxel_idx, voxel)| {
                let pos = grid.voxel_idx_to_pos(voxel_idx);
                let voxel_aabb = grid.pos_to_aabb(pos);

                let mut mapping = Vec::new();
                let mut this_voxel_aabbs = Vec::new();
                let mut this_voxel_centers = Vec::new();
                for (aabb_i, (aabb, center)) in aabbs.iter().zip(centers).enumerate() {
                    if voxel_aabb.aabb_intersect(aabb) {
                        mapping.push(aabb_i);
                        this_voxel_aabbs.push(*aabb);
                        this_voxel_centers.push(*center);
                    }
                }
                let mut bvh = Bvh2::build(&this_voxel_aabbs, &this_voxel_centers);
                if !bvh.nodes.is_empty() {
                    for i in 0..bvh.nodes.len() {
                        if bvh.nodes[i].is_leaf() {
                            let prim_index = bvh.nodes[i].first_index;
                            bvh.nodes[i].first_index = mapping[prim_index as usize] as u32;
                        }
                    }

                    // Clamp the BVH aabbs to the voxel
                    for node in &mut bvh.nodes {
                        node.aabb.min = node.aabb.min.max(voxel_aabb.min);
                        node.aabb.max = node.aabb.max.min(voxel_aabb.max);
                    }
                    *voxel = bvh;
                }
            });
        grid.voxels = voxels;
        grid
    }

    pub fn pos_to_voxel_idx(&self, pos: UVec3) -> usize {
        (pos.x + pos.y * self.resolution.x + pos.z * self.resolution.x * self.resolution.y) as usize
    }

    pub fn voxel_idx_to_pos(&self, idx: usize) -> UVec3 {
        let idx = idx as u32;
        let x = idx % self.resolution.x;
        let y = (idx / self.resolution.x) % self.resolution.y;
        let z = idx / (self.resolution.x * self.resolution.y);
        UVec3::new(x, y, z)
    }

    pub fn pos_to_aabb(&self, pos: UVec3) -> Aabb {
        let res = self.resolution.as_vec3a();
        let p1 = pos.as_vec3a() / res;
        let p2 = (pos + UVec3::ONE).as_vec3a() / res;
        let diag = self.aabb.diagonal();
        Aabb {
            min: p1 * diag + self.aabb.min,
            max: p2 * diag + self.aabb.min,
        }
    }

    pub fn world_pos_to_fvoxel_pos(&self, world_pos: Vec3A) -> Vec3A {
        ((world_pos - self.aabb.min) / self.aabb_size) * self.resolution.as_vec3a()
    }

    pub fn world_pos_to_fvoxel_pos_clamp(&self, world_pos: Vec3A) -> Vec3A {
        ((world_pos - self.aabb.min) / self.aabb_size).clamp(Vec3A::ZERO, Vec3A::ONE)
            * self.resolution.as_vec3a()
    }

    pub fn voxel_iclamp(&self, v: IVec3) -> IVec3 {
        v.clamp(IVec3::ZERO, (self.resolution - 1).as_ivec3())
    }

    pub fn traverse(&self, ray: &Ray, prims: &[Triangle]) -> Hit {
        let mut ray = *ray;
        let ires = self.resolution.as_ivec3();
        let aabb_hit = self.aabb.intersect(&ray);
        if aabb_hit > ray.tmax {
            return Hit::none();
        }

        let origin_at_voxels =
            if ray.origin.cmpge(self.aabb.min).all() && ray.origin.cmple(self.aabb.max).all() {
                ray.origin
            } else {
                ray.origin + ray.direction * aabb_hit // * 1.0001
            };

        let grid_origin = self.world_pos_to_fvoxel_pos_clamp(origin_at_voxels);

        let local_ray_direction =
            (self.world_pos_to_fvoxel_pos(origin_at_voxels + ray.direction * 100.0) - grid_origin)
                .normalize();

        let mut current_voxel = grid_origin.as_ivec3().clamp(IVec3::ZERO, ires - 1);

        let step = local_ray_direction.signum().as_ivec3();

        let t_delta = (local_ray_direction.length() / local_ray_direction).abs();

        let mut t_max = (1.0 - (grid_origin * step.as_vec3a()).fract()) * t_delta;

        let mut closest_hit = Hit::none();

        // Take one last step after the first hit is found. Needed to deal with imperfect order I'm assuming?
        // TODO Hopefully this can be removed.
        let mut last = false;
        for _ in 0..self.resolution.max_element() * 4 {
            let voxel_idx = self.pos_to_voxel_idx(current_voxel.as_uvec3());

            if !self.voxels[voxel_idx].nodes.is_empty() {
                let hit = self.voxels[voxel_idx].traverse(&ray, prims);
                // If we hit anything, we want to exit as soon as possible.
                // Since we are dda-ing in order we don't have to consider
                // possibly hitting something further along
                if hit.distance < f32::MAX {
                    if hit.distance < closest_hit.distance {
                        closest_hit = hit;
                        ray.tmax = closest_hit.distance;
                    }
                    if last {
                        break;
                    }
                    last = true;
                }
            }

            if t_max.x < t_max.y {
                if t_max.x < t_max.z {
                    current_voxel.x += step.x;
                    t_max.x += t_delta.x;
                } else {
                    current_voxel.z += step.z;
                    t_max.z += t_delta.z;
                }
            } else {
                if t_max.y < t_max.z {
                    current_voxel.y += step.y;
                    t_max.y += t_delta.y;
                } else {
                    current_voxel.z += step.z;
                    t_max.z += t_delta.z;
                }
            }

            let clamped = current_voxel.clamp(IVec3::ZERO, ires - 1);
            if clamped != current_voxel {
                return closest_hit;
            }
        }

        closest_hit
    }

    // Basic sorted traversal for testing.
    pub fn traverse2(&self, ray: &Ray, prims: &[Triangle]) -> Hit {
        let mut ray = *ray;
        let mut hit = Hit::none();

        let mut voxels = Vec::new();
        let mut voxelcenters = Vec::new();

        for i in 0..self.voxels.len() {
            let voxel_aabb = self.pos_to_aabb(self.voxel_idx_to_pos(i));
            let t = voxel_aabb.intersect(&ray);
            if t < f32::MAX {
                voxels.push((i, voxels.len()));
                voxelcenters.push(voxel_aabb.center());
            }
        }

        voxels.sort_by(|a, b| {
            voxelcenters[a.1]
                .distance(ray.origin)
                .partial_cmp(&voxelcenters[b.1].distance(ray.origin))
                .unwrap()
        });

        for i in &voxels {
            let temp_hit = self.voxels[i.0].traverse(&ray, prims);
            if temp_hit.distance < hit.distance {
                ray.tmax = temp_hit.distance;
                hit = temp_hit;
            }
        }

        hit
    }
}
