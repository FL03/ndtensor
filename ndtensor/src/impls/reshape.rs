/*
    Appellation: reshape <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{TensorBase, TensorMode};
use nd::{Data, DataOwned, Dimension, IntoDimension, RawData, ShapeArg, ShapeError};

impl<A, S, D, K> TensorBase<S, D, K>
where
    D: Dimension,
    K: TensorMode,
    S: RawData<Elem = A>,
{
    pub fn broadcast<E>(&self, shape: E) -> Option<crate::TensorView<'_, A, E::Dim, K>>
    where
        A: Clone,
        E: IntoDimension,
        S: Data,
    {
        let dim = shape.into_dimension();
        let mut ctx = self.ctx;
        ctx.set_rank(dim.ndim());
        self.data.broadcast(dim).map(|data| crate::TensorBase {
            id: self.id,
            ctx,
            data,
        })
    }
    /// Transforms the tensor into a new shape.
    pub fn into_shape<D2>(self, shape: D2) -> Result<TensorBase<S, D2::Dim, K>, ShapeError>
    where
        D2: IntoDimension,
    {
        let mut ctx = self.ctx;
        let data = self.data.into_shape(shape)?;
        ctx.set_rank(data.ndim());
        Ok(TensorBase {
            id: self.id,
            ctx,
            data,
        })
    }
    ///
    pub fn reshape<D2>(&self, shape: D2) -> Result<crate::Tensor<A, D2::Dim, K>, ShapeError>
    where
        A: Clone,
        S: Data,
        D2: ShapeArg,
    {
        self.to_shape(shape)
    }

    pub fn swap_axes(&mut self, axis1: usize, axis2: usize) {
        self.data.swap_axes(axis1, axis2);
    }
    /// Transpose the tensor.
    pub fn t(&self) -> crate::TensorView<'_, A, D, K>
    where
        A: Clone,
        S: DataOwned,
    {
        TensorBase {
            id: self.id,
            ctx: self.ctx,
            data: self.data.t(),
        }
    }

    pub fn to_shape<D2>(&self, shape: D2) -> Result<crate::Tensor<A, D2::Dim, K>, ShapeError>
    where
        A: Clone,
        S: Data,
        D2: ShapeArg,
    {
        let mut ctx = *self.ctx();
        let data = self.data.to_shape(shape)?.to_owned();
        ctx.set_rank(data.ndim());

        Ok(TensorBase {
            id: self.id,
            ctx,
            data,
        })
    }
}
