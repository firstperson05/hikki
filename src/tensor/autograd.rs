use super::Tensor;
use std::collections::HashSet;
use std::sync::{Arc, Mutex};

pub type NodeRef = Arc<Mutex<Node>>;

#[derive(Clone)]
pub enum Op {
    Leaf,
    Add(NodeRef, NodeRef),
    Matmul(NodeRef, NodeRef),
    Relu(NodeRef),
    Sigmoid(NodeRef),
    Mul(NodeRef, NodeRef), // Element-wise multiplication
}

pub struct Node {
    pub value: Tensor,
    pub grad: Option<Tensor>,
    pub op: Op,
    pub requires_grad: bool,
}

impl Node {
    pub fn new_leaf(value: Tensor, requires_grad: bool) -> NodeRef {
        Arc::new(Mutex::new(Node {
            value,
            grad: None,
            op: Op::Leaf,
            requires_grad,
        }))
    }

    pub fn add(a: &NodeRef, b: &NodeRef) -> Result<NodeRef, String> {
        let (val_a, req_a) = {
            let a_lock = a.lock().unwrap();
            (a_lock.value.clone(), a_lock.requires_grad)
        };
        let (val_b, req_b) = {
            let b_lock = b.lock().unwrap();
            (b_lock.value.clone(), b_lock.requires_grad)
        };
        let c = val_a.add(&val_b)?;

        let requires_grad = req_a || req_b;
        Ok(Arc::new(Mutex::new(Node {
            value: c,
            grad: None,
            op: Op::Add(a.clone(), b.clone()),
            requires_grad,
        })))
    }

    pub fn matmul(a: &NodeRef, b: &NodeRef) -> Result<NodeRef, String> {
        let (val_a, req_a) = {
            let a_lock = a.lock().unwrap();
            (a_lock.value.clone(), a_lock.requires_grad)
        };
        let (val_b, req_b) = {
            let b_lock = b.lock().unwrap();
            (b_lock.value.clone(), b_lock.requires_grad)
        };
        let c = val_a.matmul(&val_b)?;

        let requires_grad = req_a || req_b;
        Ok(Arc::new(Mutex::new(Node {
            value: c,
            grad: None,
            op: Op::Matmul(a.clone(), b.clone()),
            requires_grad,
        })))
    }

    pub fn relu(a: &NodeRef) -> NodeRef {
        let (val_a, req_a) = {
            let a_lock = a.lock().unwrap();
            (a_lock.value.clone(), a_lock.requires_grad)
        };
        let c = val_a.relu();

        Arc::new(Mutex::new(Node {
            value: c,
            grad: None,
            op: Op::Relu(a.clone()),
            requires_grad: req_a,
        }))
    }

    pub fn sigmoid(a: &NodeRef) -> NodeRef {
        let (val_a, req_a) = {
            let a_lock = a.lock().unwrap();
            (a_lock.value.clone(), a_lock.requires_grad)
        };
        let mut out = val_a.data.clone();
        for x in out.iter_mut() {
            *x = 1.0 / (1.0 + (-*x).exp());
        }
        let c = Tensor::new(val_a.shape, out).unwrap();
        
        Arc::new(Mutex::new(Node {
            value: c,
            grad: None,
            op: Op::Sigmoid(a.clone()),
            requires_grad: req_a,
        }))
    }

    pub fn mul(a: &NodeRef, b: &NodeRef) -> Result<NodeRef, String> {
        let (val_a, req_a) = {
            let a_lock = a.lock().unwrap();
            (a_lock.value.clone(), a_lock.requires_grad)
        };
        let (val_b, req_b) = {
            let b_lock = b.lock().unwrap();
            (b_lock.value.clone(), b_lock.requires_grad)
        };
        if val_a.shape != val_b.shape {
            return Err("Shape mismatch in mul".to_string());
        }
        let mut out = vec![0.0; val_a.data.len()];
        for i in 0..out.len() {
            out[i] = val_a.data[i] * val_b.data[i];
        }
        let c = Tensor::new(val_a.shape, out)?;
        let requires_grad = req_a || req_b;
        Ok(Arc::new(Mutex::new(Node {
            value: c,
            grad: None,
            op: Op::Mul(a.clone(), b.clone()),
            requires_grad,
        })))
    }

    pub fn backward(node: NodeRef) {
        let mut topo = Vec::new();
        let mut visited = HashSet::new();

        fn build_topo(n: &NodeRef, topo: &mut Vec<NodeRef>, visited: &mut HashSet<usize>) {
            let ptr = Arc::as_ptr(n) as usize;
            if !visited.contains(&ptr) {
                visited.insert(ptr);
                let n_lock = n.lock().unwrap();
                let op_clone = n_lock.op.clone();
                drop(n_lock); // Release lock before recursion
                
                match op_clone {
                    Op::Leaf => {}
                    Op::Add(a, b) | Op::Mul(a, b) | Op::Matmul(a, b) => {
                        build_topo(&a, topo, visited);
                        build_topo(&b, topo, visited);
                    }
                    Op::Relu(a) | Op::Sigmoid(a) => {
                        build_topo(&a, topo, visited);
                    }
                }
                topo.push(n.clone());
            }
        }

        build_topo(&node, &mut topo, &mut visited);

        // Initialize gradient for output node
        {
            let mut n_mut = node.lock().unwrap();
            if n_mut.grad.is_none() {
                let shape = n_mut.value.shape.clone();
                n_mut.grad = Some(Tensor::new(shape, vec![1.0; n_mut.value.numel()]).unwrap());
            }
        }

        for n in topo.into_iter().rev() {
            let (op, grad) = {
                let n_lock = n.lock().unwrap();
                (n_lock.op.clone(), n_lock.grad.clone())
            };

            let g = match grad {
                Some(g) => g,
                None => continue,
            };

            match op {
                Op::Add(a, b) => {
                    Self::accumulate_grad(&a, &g);
                    Self::accumulate_grad(&b, &g);
                }
                Op::Matmul(a, b) => {
                    // a_grad = g x b^T
                    // b_grad = a^T x g
                    let a_req_grad = {
                        let a_lock = a.lock().unwrap();
                        a_lock.requires_grad
                    };
                    if a_req_grad {
                        let b_val = {
                            let b_lock = b.lock().unwrap();
                            b_lock.value.clone()
                        };
                        let b_t = transpose2d(&b_val).unwrap();
                        let a_g = g.matmul(&b_t).unwrap();
                        Self::accumulate_grad(&a, &a_g);
                    }
                    let b_req_grad = {
                        let b_lock = b.lock().unwrap();
                        b_lock.requires_grad
                    };
                    if b_req_grad {
                        let a_val = {
                            let a_lock = a.lock().unwrap();
                            a_lock.value.clone()
                        };
                        let a_t = transpose2d(&a_val).unwrap();
                        let b_g = a_t.matmul(&g).unwrap();
                        Self::accumulate_grad(&b, &b_g);
                    }
                }
                Op::Relu(a) => {
                    let a_req_grad = {
                        let a_lock = a.lock().unwrap();
                        a_lock.requires_grad
                    };
                    if a_req_grad {
                        let a_val = {
                            let a_lock = a.lock().unwrap();
                            a_lock.value.clone()
                        };
                        let mut new_g = g.clone();
                        for i in 0..new_g.data.len() {
                            if a_val.data[i] <= 0.0 {
                                new_g.data[i] = 0.0;
                            }
                        }
                        Self::accumulate_grad(&a, &new_g);
                    }
                }
                Op::Sigmoid(a) => {
                    let a_req_grad = {
                        let a_lock = a.lock().unwrap();
                        a_lock.requires_grad
                    };
                    if a_req_grad {
                        let s_val = {
                            let n_lock = n.lock().unwrap();
                            n_lock.value.clone()
                        };
                        let mut new_g = g.clone();
                        let s = &s_val.data;
                        for i in 0..new_g.data.len() {
                            new_g.data[i] *= s[i] * (1.0 - s[i]);
                        }
                        Self::accumulate_grad(&a, &new_g);
                    }
                }
                Op::Mul(a, b) => {
                    let a_req_grad = {
                        let a_lock = a.lock().unwrap();
                        a_lock.requires_grad
                    };
                    if a_req_grad {
                        let b_val = {
                            let b_lock = b.lock().unwrap();
                            b_lock.value.clone()
                        };
                        let mut a_g = g.clone();
                        for i in 0..a_g.data.len() {
                            a_g.data[i] *= b_val.data[i];
                        }
                        Self::accumulate_grad(&a, &a_g);
                    }
                    let b_req_grad = {
                        let b_lock = b.lock().unwrap();
                        b_lock.requires_grad
                    };
                    if b_req_grad {
                        let a_val = {
                            let a_lock = a.lock().unwrap();
                            a_lock.value.clone()
                        };
                        let mut b_g = g.clone();
                        for i in 0..b_g.data.len() {
                            b_g.data[i] *= a_val.data[i];
                        }
                        Self::accumulate_grad(&b, &b_g);
                    }
                }
                Op::Leaf => {}
            }
        }
    }

    fn accumulate_grad(node: &NodeRef, grad: &Tensor) {
        let mut n = node.lock().unwrap();
        if !n.requires_grad {
            return;
        }
        if let Some(ref mut ext_grad) = n.grad {
            *ext_grad = ext_grad.add(grad).unwrap();
        } else {
            n.grad = Some(grad.clone());
        }
    }
}

// 2D Matrix Transpose helper
pub fn transpose2d(t: &Tensor) -> Result<Tensor, String> {
    if t.shape.len() != 2 {
        return Err("transpose2d expects 2D tensor".to_string());
    }
    let rows = t.shape[0];
    let cols = t.shape[1];
    let mut out = vec![0.0; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = t.data[r * cols + c];
        }
    }
    Tensor::new(vec![cols, rows], out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_autograd_basic() {
        let a = Node::new_leaf(
            Tensor::new(vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]).unwrap(),
            true,
        );
        let b = Node::new_leaf(
            Tensor::new(vec![2, 2], vec![2.0, 0.0, 1.0, 2.0]).unwrap(),
            true,
        );
        let c = Node::matmul(&a, &b).unwrap();
        let d = Node::add(&c, &a).unwrap();

        Node::backward(d);
        assert!(a.lock().unwrap().grad.is_some());
        assert!(b.lock().unwrap().grad.is_some());
    }
}
