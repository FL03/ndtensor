/*
    Appellation: ops <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{expr::*, wrapper::TensorOp};

pub(crate) mod expr;
pub(crate) mod wrapper;

use nd::RawData;

pub type TOp<A, B> = TensorOp<nd::OwnedArcRepr<A>, nd::OwnedArcRepr<B>>;

pub trait NdTensorOp<S1, S2>
where
    S1: RawData,
    S2: RawData,
{
    fn is_none(&self) -> bool;
    fn is_some(&self) -> bool;

    fn as_ref(&self) -> Option<&TensorExpr<S1, S2>>;

    fn as_mut(&mut self) -> Option<&mut TensorExpr<S1, S2>>;
}

impl<A, B, S1, S2> NdTensorOp<S1, S2> for Option<TensorExpr<S1, S2>>
where
    S1: RawData<Elem = A>,
    S2: RawData<Elem = B>,
{
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
