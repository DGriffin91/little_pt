use std::array;

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
