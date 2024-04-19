/*
    Appellation: dimensional <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::ops::TensorOp;
use crate::{Tensor, TensorBase, TensorView, TensorViewMut};

use nd::{Data, DataMut, RawData};
use nd::{Ix, Ix0, Ix1, Ix2};

impl<A> Tensor<A, Ix0> {
    pub fn into_scalar(self) -> A {
        self.data.into_scalar()
    }
}

impl<A, S> TensorBase<S, Ix2>
where
    S: RawData<Elem = A>,
{
    pub fn column(&self, idx: Ix) -> TensorView<'_, A, Ix1>
    where
        S: Data<Elem = A>,
    {
        let data = self.data.column(idx);
        TensorBase::new(data, None, self.ctx.is_variable())
    }

    pub fn column_mut(&mut self, idx: Ix) -> TensorViewMut<'_, A, Ix1>
    where
        S: DataMut<Elem = A>,
    {
        let ctx = *self.ctx();
        let data = self.data.column_mut(idx);
        TensorBase {
            id: self.id,
            ctx,
            data,
            op: TensorOp::none(),
        }
    }

    pub fn is_square(&self) -> bool {
        self.data().is_square()
    }

    pub fn ncols(&self) -> usize {
        self.data().ncols()
    }

    pub fn nrows(&self) -> usize {
        self.data().nrows()
    }

    pub fn row(&self, idx: Ix) -> TensorView<'_, A, Ix1>
    where
        S: Data<Elem = A>,
    {
        let data = self.data.row(idx);
        TensorBase::new(data, None, self.ctx.is_variable())
    }

    pub fn row_mut(&mut self, idx: Ix) -> TensorViewMut<'_, A, Ix1>
    where
        S: DataMut<Elem = A>,
    {
        let data = self.data.row_mut(idx);
        TensorBase::new(data, None, self.ctx.is_variable())
    }
}
