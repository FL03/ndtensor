/*
    Appellation: numerical <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{TensorBase, TensorMode};
use core::marker::PhantomData;
use nd::{Data, Dimension};
use num::traits::{NumCast, ToPrimitive};

impl<A, S, D, K> TensorBase<S, D, K>
where
    A: ToPrimitive,
    D: Dimension,
    K: TensorMode,
    S: Data<Elem = A>,
{
    pub fn numcast<B>(&self) -> crate::Tensor<B, D, K>
    where
        A: Clone,
        B: NumCast,
    {
        TensorBase {
            id: self.id,
            ctx: self.ctx,
            data: self.data.mapv(|x| B::from(x).unwrap()),
            op: self.op.numcast(),
            _kind: PhantomData::<K>,
        }
    }
}
