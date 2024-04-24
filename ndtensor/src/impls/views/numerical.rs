/*
    Appellation: numerical <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::TensorBase;
use nd::{Data, Dimension};
use num::traits::{NumCast, ToPrimitive};

impl<A, S, D> TensorBase<S, D>
where
    A: ToPrimitive,
    D: Dimension,
    S: Data<Elem = A>,
{
    pub fn numcast<B>(&self) -> crate::Tensor<B, D>
    where
        A: Clone,
        B: NumCast,
    {
        TensorBase {
            id: self.id,
            ctx: self.ctx,
            data: self.data().mapv(|x| B::from(x).unwrap()),
            op: self.op.numcast(),
        }
    }
}
