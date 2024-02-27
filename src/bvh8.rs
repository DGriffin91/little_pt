use std::array;

use glam::{vec3a, Vec3A};

use crate::{
    aabb::Aabb,
    bvh2::{Bvh2, Bvh2Node},
    triangle::Triangle,
    Hit, Ray,
};

pub const BRANCHING: usize = 8;

#[derive(Default, Clone, Debug)]
pub struct Child {
    pub bounds: Aabb,
    pub node: Option<Box<Bvh8Node>>,
}

#[derive(Default, Clone, Debug)]
pub struct Leaf {
    pub bounds: Aabb,
    pub prims: [u32; 3],
}

#[derive(Clone, Debug)]
pub enum Bvh8Node {
    Inner([Child; BRANCHING]),
    Leaf(Leaf),
}

impl Bvh8Node {
    pub fn empty() -> Bvh8Node {
        Bvh8Node::Inner(array::from_fn(|_| Child::default()))
    }

    pub fn from_bvh2(bvh2: &Bvh2) -> Self {
        let mut node = Bvh8Node::empty();
        Bvh8Node::collapse_bvh2(bvh2, 0, &mut node);
        node
    }

    fn collapse_bvh2(bvh2: &Bvh2, bvh2_node_index: u32, node: &mut Bvh8Node) {
        let top_node = &bvh2.nodes[bvh2_node_index as usize];

        if top_node.is_leaf() {
            *node = convert_leaf(top_node, bvh2);
            return;
        }

        // Collect up to 8 child nodes in Bvh2 format
        let mut nodes8 = Vec::new();

        let (node0_a, node0_b) = bvh2_get_children(bvh2, top_node);

        if node0_a.0.is_leaf() {
            nodes8.push(node0_a);
        } else {
            let (node1_a, node1_b) = bvh2_get_children(bvh2, node0_a.0);
            if node1_a.0.is_leaf() {
                nodes8.push(node1_a);
            } else {
                let (node2_a, node2_b) = bvh2_get_children(bvh2, node1_a.0);
                nodes8.push(node2_a);
                nodes8.push(node2_b);
            }
            if node1_b.0.is_leaf() {
                nodes8.push(node1_b);
            } else {
                let (node2_a, node2_b) = bvh2_get_children(bvh2, node1_b.0);
                nodes8.push(node2_a);
                nodes8.push(node2_b);
            }
        }

        if node0_b.0.is_leaf() {
            nodes8.push(node0_b);
        } else {
            let (node1_a, node1_b) = bvh2_get_children(bvh2, node0_b.0);
            if node1_a.0.is_leaf() {
                nodes8.push(node1_a);
            } else {
                let (node2_a, node2_b) = bvh2_get_children(bvh2, node1_a.0);
                nodes8.push(node2_a);
                nodes8.push(node2_b);
            }
            if node1_b.0.is_leaf() {
                nodes8.push(node1_b);
            } else {
                let (node2_a, node2_b) = bvh2_get_children(bvh2, node1_b.0);
                nodes8.push(node2_a);
                nodes8.push(node2_b);
            }
        }

        let mut children: [Child; 8] = array::from_fn(|_| Child::default());

        let mut nodes8_iter = nodes8.iter();

        // Recursively convert them to Node8's
        for child in &mut children {
            if let Some(node) = nodes8_iter.next() {
                if node.0.is_leaf() {
                    *child = Child {
                        bounds: node.0.aabb,
                        node: Some(Box::new(convert_leaf(node.0, bvh2))),
                    }
                } else {
                    let mut tmp_node8 = Bvh8Node::empty();
                    Bvh8Node::collapse_bvh2(bvh2, node.1, &mut tmp_node8);
                    *child = Child {
                        bounds: node.0.aabb,
                        node: Some(Box::new(tmp_node8)),
                    };
                }
            }
        }

        *node = Bvh8Node::Inner(children);
    }

    pub fn order_subtree(&mut self, self_bounds: &Aabb) {
        if let Bvh8Node::Inner(children) = self {
            Bvh8Node::order_children(children, self_bounds);
            for child in children {
                if let Some(node) = child.node.as_mut() {
                    node.order_subtree(&child.bounds);
                }
            }
        }
    }

    // Based on https://github.com/jan-van-bergen/GPU-Raytracer/blob/33896a93c3772b8f81719a9b4441f44f87a4a50e/Src/BVH/Builders/CWBVHBuilder.cpp#L155
    fn order_children(children: &mut [Child; 8], self_bounds: &Aabb) {
        let p = self_bounds.center();

        let mut cost = [[0.0_f32; 8]; BRANCHING];

        // Corresponds directly to the number of bit patterns we're creating
        const DIRECTIONS: usize = 8;

        // Fill cost table
        for (c, child) in children.iter().enumerate() {
            for s in 0..DIRECTIONS {
                let direction = vec3a(
                    if (s & 0b100) != 0 { -1.0 } else { 1.0 },
                    if (s & 0b010) != 0 { -1.0 } else { 1.0 },
                    if (s & 0b001) != 0 { -1.0 } else { 1.0 },
                );

                cost[c][s] = Vec3A::dot(child.bounds.center() - p, direction);
            }
        }

        const INVALID: u32 = !0;

        let mut assignment = [INVALID; BRANCHING];
        let mut slot_filled = [false; DIRECTIONS];

        // The paper suggests the auction method, but greedy is almost as good.
        loop {
            let mut min_cost = f32::MAX;

            let mut min_slot = INVALID;
            let mut min_index = INVALID;

            // Find cheapest unfilled slot of any unassigned child
            for c in 0..children.len() {
                if assignment[c] == INVALID {
                    for (s, &slot_filled) in slot_filled.iter().enumerate() {
                        if !slot_filled && cost[c][s] < min_cost {
                            min_cost = cost[c][s];

                            min_slot = s as _;
                            min_index = c as _;
                        }
                    }
                }
            }

            if min_slot == INVALID {
                break;
            }

            slot_filled[min_slot as usize] = true;
            assignment[min_index as usize] = min_slot;
        }

        // Permute children array according to assignment
        let original_order = std::mem::replace(children, array::from_fn(|_| Child::default()));

        let mut child_assigned = [false; BRANCHING];
        for (assignment, new_value) in assignment.into_iter().zip(original_order.into_iter()) {
            children[assignment as usize] = new_value;
            child_assigned[assignment as usize] = true;
        }
        assert_eq!(child_assigned, [true; BRANCHING]);
    }

    pub fn traverse_recursive(&self, ray: &Ray, prims: &[Triangle], hit: &mut Hit) {
        match self {
            Bvh8Node::Inner(children) => {
                for child in children {
                    if child.bounds.intersect(ray) < hit.distance {
                        if let Some(node) = &child.node {
                            node.traverse_recursive(ray, prims, hit)
                        }
                    }
                }
            }
            Bvh8Node::Leaf(leaf) => {
                for prim_index in leaf.prims {
                    if prim_index == u32::MAX {
                        continue;
                    }
                    let (hit_dist, uv) = prims[prim_index as usize].intersect(ray);
                    if hit_dist < hit.distance {
                        hit.prim_index = prim_index;
                        hit.distance = hit_dist;
                        hit.uv = uv;
                    }
                }
            }
        }
    }
}

fn convert_leaf(bvh2_node: &Bvh2Node, bvh2: &Bvh2) -> Bvh8Node {
    let mut prims = [u32::MAX; 3];
    for i in 0..bvh2_node.prim_count as usize {
        prims[i] = bvh2.prim_indices[bvh2_node.first_index as usize + i];
    }
    Bvh8Node::Leaf(Leaf {
        bounds: bvh2_node.aabb,
        prims,
    })
}

fn bvh2_get_children<'a>(
    bvh2: &'a Bvh2,
    top_node: &'a Bvh2Node,
) -> ((&'a Bvh2Node, u32), (&'a Bvh2Node, u32)) {
    let idx_a = top_node.first_index;
    let idx_b = top_node.first_index + 1;
    (
        (&bvh2.nodes[idx_a as usize], idx_a),
        (&bvh2.nodes[idx_b as usize], idx_b),
    )
}
