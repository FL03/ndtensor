/*
    Appellation: reshape <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::ops::TensorExpr;
use crate::TensorBase;
use nd::{Data, DataOwned, Dimension, IntoDimension, RawData, ShapeArg, ShapeError};

impl<A, S, D> TensorBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// Transforms the tensor into a new shape.
    pub fn into_shape<D2>(self, shape: D2) -> Result<TensorBase<S, D2::Dim>, ShapeError>
    where
        D2: IntoDimension,
    {
        let data = self.data.into_shape(shape)?;
        Ok(TensorBase {
            id: self.id,
            ctx: self.ctx,
            data,
            op: self.op,
        })
    }
    ///
    pub fn reshape<D2>(&self, shape: D2) -> Result<crate::Tensor<A, D2::Dim>, ShapeError>
    where
        A: Clone,
        S: Data,
        D2: ShapeArg,
    {
        self.to_shape(shape)
    }
    /// Transpose the tensor.
    pub fn t(&self) -> crate::Tensor<A, D>
    where
        A: Clone,
        S: DataOwned,
    {
        TensorBase {
            id: self.id,
            ctx: self.ctx,
            data: self.data().t().to_owned(),
            op: TensorExpr::transpose(self.to_owned().into_dyn().boxed()).into(),
        }
    }

    pub fn to_shape<D2>(&self, shape: D2) -> Result<crate::Tensor<A, D2::Dim>, ShapeError>
    where
        A: Clone,
        S: Data,
        D2: ShapeArg,
    {
        let data = self.data.to_shape(shape)?.to_owned();
        Ok(TensorBase {
            id: self.id,
            ctx: self.ctx,
            data,
            op: self.op.to_owned(),
        })
    }
}
