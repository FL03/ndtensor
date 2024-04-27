/*
    Appellation: grad <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::{TensorError, TensorExpr, TensorGrad, TensorId};
use crate::TensorBase;
use acme::ops::{Arithmetic, BinaryOp, UnaryOp};
use acme::prelude::Scalar;
use nd::{Data, Dimension, OwnedRepr, RawDataClone, ScalarOperand};
// use num::complex::ComplexFloat;
use std::collections::HashMap;

macro_rules! entry {
    ($ctx:expr, $entry:expr) => {
        entry!($ctx, $entry, zeros_like)
    };
    ($ctx:expr, $entry:expr, $call:ident) => {
        entry!($ctx, $entry, $entry.$call())
    };
    ($ctx:expr, $entry:expr, $default:expr) => {
        $ctx.entry(*$entry.id()).or_insert($default)
    };
}

impl<A, S, D> TensorBase<S, D>
where
    A: Scalar + ScalarOperand,
    D: Dimension,
    S: Data<Elem = A> + RawDataClone,
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

    /// [backward](TensorBase::backward) is a function which computes the gradient of the tensor with respect to each variable.
    pub fn backward(&self) -> Result<TensorGrad<OwnedRepr<A>>, TensorError>
    where
        A: Scalar<Real = A>,
    {
        // get the sorted nodes
        let sorted = self.toposort(true);
        // initialize a new gradient store
        let mut store = TensorGrad::new();
        // insert the gradient w.r.t. the current node
        store.or_insert_ones(&self.to_owned().into_dyn());

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
                    TensorExpr::Binary { lhs, rhs, op } => {
                        if rhs.is_scalar() {
                            let rhs = rhs.to_owned().into_dimensionality::<nd::Ix0>().unwrap();
                            let val = rhs.into_scalar();
                            match op {
                                BinaryOp::Arith(inner) => match inner {
                                    Arithmetic::Add(_) => {
                                        *entry!(store, lhs) += &grad;
                                    }
                                    Arithmetic::Div(_) => {
                                        *entry!(store, lhs) += &grad.div_scalar(val);
                                    }
                                    Arithmetic::Mul(_) => {
                                        *entry!(store, lhs) += &grad.mul_scalar(val);
                                    }
                                    Arithmetic::Pow(_) => {
                                        *entry!(store, lhs) += &grad.mul(
                                            &lhs.powf(val - A::from(1).unwrap()).mul_scalar(val),
                                        );
                                    }
                                    Arithmetic::Sub(_) => {
                                        *entry!(store, lhs) += &grad;
                                    }
                                    _ => todo!(),
                                },
                                _ => todo!(),
                            }
                        } else {
                            match op {
                                BinaryOp::Arith(inner) => match inner {
                                    Arithmetic::Add(_) => {
                                        *entry!(store, lhs) += &grad;
                                        *entry!(store, rhs) += &grad;
                                    }
                                    Arithmetic::Div(_) => {
                                        *entry!(store, lhs) += &grad.div(&rhs);
                                        *entry!(store, rhs) -= &grad.mul(&lhs.div(&rhs.powi(2)));
                                    }
                                    Arithmetic::Mul(_) => {
                                        *entry!(store, lhs) += &grad.mul(&rhs);
                                        *entry!(store, rhs) += &grad.mul(&lhs);
                                    }
                                    Arithmetic::Sub(_) => {
                                        *entry!(store, lhs) += &grad;
                                        *entry!(store, rhs) -= &grad;
                                    }
                                    _ => todo!(),
                                },
                                _ => todo!(),
                            }
                        }
                    }
                    TensorExpr::Unary { recv, op } => match op {
                        UnaryOp::Cos => {
                            *entry!(store, recv) += &grad.mul(&recv.sin().neg());
                        }
                        UnaryOp::Cosh => {
                            *entry!(store, recv) += &grad.mul(&recv.sinh());
                        }
                        UnaryOp::Exp => {
                            *entry!(store, recv) += &grad.mul(&recv.exp());
                        }
                        UnaryOp::Ln => {
                            *entry!(store, recv) += &grad.div(&recv);
                        }
                        UnaryOp::Neg => {
                            *entry!(store, recv) -= &grad;
                        }
                        UnaryOp::Sin => {
                            *entry!(store, recv) += &grad.mul(&recv.cos());
                        }
                        UnaryOp::Sinh => {
                            *entry!(store, recv) += &grad.mul(&recv.cosh());
                        }
                        UnaryOp::Square => {
                            *entry!(store, recv) +=
                                &grad.mul(&recv.mul_scalar(A::from(2).unwrap()));
                        }
                        UnaryOp::Sqrt => {
                            *entry!(store, recv) +=
                                &grad.div(&recv.sqrt().mul_scalar(A::from(2).unwrap()));
                        }
                        UnaryOp::Tan => {
                            *entry!(store, recv) +=
                                &grad.mul(&recv.ones_like().div(&recv.cos().powi(2)));
                        }
                        UnaryOp::Tanh => {
                            *entry!(store, recv) += &grad
                                .mul(&recv.tanh().powi(2).neg().add_scalar(A::from(1).unwrap()));
                        }
                        _ => todo!(),
                    },
                    _ => todo!(),
                }
            }
        }
        Ok(store)
    }
    /// Compute the gradient of the tensor w.r.t. a particular variable (tensor)
    pub fn grad(&self, target: &TensorId) -> Result<crate::Tensor<A>, TensorError>
    where
        A: Scalar<Real = A>,
    {
        let store = self.backward()?;
        let grad = store.get(target).expect("Gradient not found");
        Ok(grad.to_owned())
    }
}
