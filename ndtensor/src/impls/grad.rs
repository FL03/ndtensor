/*
    Appellation: grad <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::{TensorError, TensorExpr, TensorGrad, TensorId};
use crate::TensorBase;
use acme::ops::{Arithmetic, BinaryOp, UnaryOp};
use acme::prelude::Scalar;
use ndarray::{Data, DataOwned, OwnedRepr, RawDataClone};
// use num::complex::ComplexFloat;
use std::collections::HashMap;

pub(crate) type Visited<K = TensorId> = HashMap<K, bool>;

macro_rules! entry {
    ($ctx:expr, $entry:expr) => {
        entry!($ctx, $entry, $entry.zeros_like())
    };
    ($ctx:expr, $entry:expr, $default:expr) => {
        $ctx.entry($entry.id()).or_insert($default)
    };
}

impl<S> TensorBase<S>
where
    S: Data + RawDataClone,
{
    /// toposort is a function which sorts the nodes of the op graph in topological order.
    fn toposort(&self, reverse: bool) -> Vec<TensorBase<S>> {
        let scope = self.to_dyn();

        let (_tg, mut nodes) = crate::walk(scope, Vec::new(), &mut HashMap::new());
        if reverse {
            nodes.reverse();
        }
        nodes
    }
}

impl<A, S> TensorBase<S>
where
    A: Scalar,
    S: Data<Elem = A> + DataOwned + RawDataClone,
{
    /// grad is a function which computes the gradient of the tensor with respect to the input tensor.
    pub fn grad(&self) -> Result<TensorGrad<OwnedRepr<A>>, TensorError> {
        // get the sorted nodes
        let sorted = self.toposort(true);
        // initialize a new gradient store
        let mut store = TensorGrad::new();
        // insert the gradient w.r.t. the current node
        store.or_insert_ones(&self.to_owned());

        for node in sorted.iter() {
            if node.is_variable() {
                continue;
            }
            // get the gradient of the node
            let grad = store
                .remove_item(&node.to_owned())
                .expect("Gradient not found");
            // detach the gradient
            let grad = grad.detach().to_owned();
            // handle the different types of operations
            if let Some(expr) = node.op() {
                let expr = expr.to_owned();
                match expr {
                    TensorExpr::Binary { lhs, rhs, op } => match op {
                        BinaryOp::Arithmetic(inner) => match inner {
                            Arithmetic::Add(_) => {
                                *entry!(store, lhs) += &grad;
                                *entry!(store, rhs) += &grad;
                            }
                            // Arithmetic::Div(_) => {
                            //     *entry!(store, lhs) += grad.iter().zip(rhs.iter()).map(|(l, r)| l / r).collect();
                            //     *entry!(store, rhs) -= &grad * &lhs / rhs.powi(2);
                            // },
                            // Arithmetic::Mul(_) => {
                            //     *entry!(store, lhs) += &grad * &rhs;
                            //     *entry!(store, rhs) += &grad * &lhs;
                            // },
                            // Arithmetic::Sub(_) => {
                            //     *entry!(store, lhs) += &grad;
                            //     *entry!(store, rhs) -= &grad;
                            // },
                            _ => todo!(),
                        },
                        _ => todo!(),
                    },
                    TensorExpr::Unary { recv, op } => match op {
                        _ => todo!(),
                    },
                    _ => todo!(),
                }
            }
        }
        Ok(store)
    }
}
