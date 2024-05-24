/*
    Appellation: context <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::TensorBase;
use acme::ops::{BinaryOp, UnaryOp};
use nd::*;

pub struct Context<A> {
    pub expr: Expr<A>,
    pub vars: Vec<String>,
}

pub enum Expr<A, S = OwnedRepr<A>, D = nd::IxDyn>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    Binary {
        lhs: Box<Expr<A, S, D>>,
        op: BinaryOp,
        rhs: Box<Expr<A, S, D>>,
    },
    Unary {
        recv: Box<Expr<A, S, D>>,
        op: UnaryOp,
    },
    Scalar(A),
    Tensor(TensorBase<S, D>),
}
