use crate::{aabb::Aabb, triangle::Triangle, Hit, Ray};
use glam::Vec3A;

#[derive(Default, Clone)]
pub struct Node {
    pub aabb: Aabb,
    pub prim_count: u32, // Not needed for rendering. Could make a smaller node just for rendering.
    pub first_index: u32,
}

impl Node {
    pub fn new(aabb: Aabb, prim_count: u32, first_index: u32) -> Self {
        Self {
            aabb,
            prim_count,
            first_index,
        }
    }

    #[inline]
    pub fn is_leaf(&self) -> bool {
        self.prim_count != 0
    }
}

#[derive(Clone, Default)]
pub struct Bvh {
    pub nodes: Vec<Node>,
}

impl Bvh {
    pub fn depth(&self, node_index: usize) -> usize {
        let node = &self.nodes[node_index];
        if node.is_leaf() {
            1
        } else {
            1 + self
                .depth(node.first_index as usize)
                .max(self.depth((node.first_index + 1) as usize))
        }
    }
}

pub fn build_recursive(
    bvh: &mut Bvh,
    node_index: usize,
    node_count: &mut usize,
    prim_indices: &mut [usize],
    aabbs: &[Aabb],
    centers: &[Vec3A],
) {
    let mut node = bvh.nodes[node_index].clone();
    assert!(node.is_leaf());

    node.aabb = Aabb::empty();
    let mut centers_aabb = Aabb::empty();
    let first_index = node.first_index as usize;
    for i in 0..node.prim_count as usize {
        let prim_index = prim_indices[first_index + i];
        node.aabb = node.aabb.union(&aabbs[prim_index]);
        centers_aabb.extend(centers[prim_index]);
    }

    let centers_lg_axis = centers_aabb.largest_axis();

    if node.prim_count <= 1 {
        bvh.nodes[node_index] = node;
        return;
    }

    // Global index of the first primitive in the right child
    let first_right = if node.prim_count > 1 {
        // Median split

        // was node.aabb.largest_axis(), centers_lg_axis seems faster,
        // at least with primary rays on sponza
        let axis = centers_lg_axis;

        // Sort the primitive indices in place
        let prim_indices = &mut prim_indices[first_index..first_index + node.prim_count as usize];
        prim_indices
            .sort_unstable_by(|&i, &j| centers[i][axis].partial_cmp(&centers[j][axis]).unwrap());
        // put first_right half way though set of indices
        first_index + node.prim_count as usize / 2
    } else {
        // Terminate with a leaf.
        // Keeps all the child primitives in this leaf, this branch ends here.
        bvh.nodes[node_index] = node;
        return;
    };

    let first_child = *node_count;
    *node_count += 2;

    bvh.nodes[first_child].prim_count = first_right as u32 - first_index as u32;
    bvh.nodes[first_child + 1].prim_count = node.prim_count - bvh.nodes[first_child].prim_count;
    bvh.nodes[first_child].first_index = first_index as u32;
    bvh.nodes[first_child + 1].first_index = first_right as u32;

    node.first_index = first_child as u32;
    node.prim_count = 0;

    bvh.nodes[node_index] = node;

    build_recursive(bvh, first_child, node_count, prim_indices, aabbs, centers);
    build_recursive(
        bvh,
        first_child + 1,
        node_count,
        prim_indices,
        aabbs,
        centers,
    );
}

impl Bvh {
    pub fn build(aabbs: &[Aabb], centers: &[Vec3A]) -> Bvh {
        let mut prim_indices: Vec<usize> = (0..aabbs.len()).collect();
        let mut bvh = Bvh {
            nodes: Vec::with_capacity((2 * aabbs.len() as i64 - 1).max(0) as usize),
        };

        let prim_count = (aabbs.len()) as u32;

        bvh.nodes
            .resize((2 * prim_count as i64 - 1).max(0) as usize, Node::default());

        if bvh.nodes.is_empty() {
            bvh
        } else {
            bvh.nodes[0] = Node {
                aabb: Aabb::empty(),
                prim_count,
                first_index: 0,
            };

            let mut node_count = 1;
            build_recursive(
                &mut bvh,
                0,
                &mut node_count,
                &mut prim_indices,
                aabbs,
                centers,
            );
            bvh.nodes.resize(node_count, Node::default());

            for i in 0..bvh.nodes.len() {
                if bvh.nodes[i].is_leaf() {
                    bvh.nodes[i].first_index =
                        prim_indices[bvh.nodes[i].first_index as usize] as u32;
                }
            }

            bvh
        }
    }

    pub fn traverse(&self, ray: &Ray, prims: &[Triangle]) -> Hit {
        let mut hit = Hit::none();
        let mut stack = Vec::new();
        stack.push(0);
        let mut min_dist = ray.tmax;
        while let Some(current_node_index) = stack.pop() {
            let node = &self.nodes[current_node_index];
            if node.aabb.ray_intersect(ray) >= min_dist {
                continue;
            }

            if node.is_leaf() {
                let prim_index = node.first_index as usize;
                let (hit_dist, uv) = prims[prim_index].intersect(ray);
                if hit_dist < min_dist {
                    hit.prim_index = prim_index as u32;
                    hit.distance = hit_dist;
                    hit.uv = uv;
                    min_dist = hit_dist;
                }
            } else {
                stack.push(node.first_index as usize);
                stack.push((node.first_index + 1) as usize);
            }
        }

        hit
    }

    pub fn traverse_recursive(&self, ray: &Ray, node_index: usize, indices: &mut Vec<usize>) {
        let node = &self.nodes[node_index];

        if node.is_leaf() {
            let prim_index = node.first_index as usize;
            indices.push(prim_index);
        } else {
            if node.aabb.ray_intersect(ray) < f32::MAX {
                self.traverse_recursive(ray, node.first_index as usize, indices);
                self.traverse_recursive(ray, node.first_index as usize + 1, indices);
            }
        }
    }
}
