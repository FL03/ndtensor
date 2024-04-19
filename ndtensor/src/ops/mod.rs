/*
    Appellation: ops <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{expr::*, wrapper::TensorOp};

pub(crate) mod expr;
pub(crate) mod wrapper;

use nd::*;

pub type TOp<A, B> = TensorOp<nd::OwnedArcRepr<A>, nd::OwnedArcRepr<B>>;

pub trait NdTensorOp<S1, S2>
where
    S1: RawData,
    S2: RawData,
{
    fn from_expr(expr: TensorExpr<S1, S2>) -> Self;
    fn new(expr: Option<TensorExpr<S1, S2>>) -> Self;
    fn none() -> Self;

    fn is_none(&self) -> bool;
    fn is_some(&self) -> bool;

    fn as_ref(&self) -> Option<&TensorExpr<S1, S2>>;

    fn as_mut(&mut self) -> Option<&mut TensorExpr<S1, S2>>;
}

pub trait NdTensorOpExt<S1, S2>: NdTensorOp<S1, S2> where S1: RawData, S2: RawData {
    type Op: NdTensorOp<S1, S2>;

    fn map<F, O, T1>(self, f: F) -> O
    where
        O: NdTensorOp<T1, T1> + Sized,
        F: FnOnce(TensorExpr<S1, S2>) -> TensorExpr<T1>,
        T1: RawData;
}


impl<A, B, S1, S2> NdTensorOp<S1, S2> for Option<TensorExpr<S1, S2>>
where
    S1: RawData<Elem = A>,
    S2: RawData<Elem = B>,
{
    fn from_expr(expr: TensorExpr<S1, S2>) -> Self {
        Some(expr)
    }

    fn new(expr: Option<TensorExpr<S1, S2>>) -> Self {
        expr
    }

    fn none() -> Self {
        None
    }

    fn is_none(&self) -> bool {
        self.is_none()
    }

    fn is_some(&self) -> bool {
        self.is_some()
    }

    fn as_ref(&self) -> Option<&TensorExpr<S1, S2>> {
        self.as_ref()
    }

    fn as_mut(&mut self) -> Option<&mut TensorExpr<S1, S2>> {
        self.as_mut()
    }
}
