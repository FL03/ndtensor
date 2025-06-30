/*
    appellation: impl_tensor_iter <module>
    authors: @FL03
*/
use crate::tensor::TensorBase;

use crate::iter::{Iter, IterMut};
use ndarray::{Data, DataMut, Dimension, RawData};

impl<A, S, D> TensorBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// returns an immutable iterator over the elements of the tensor
    pub fn iter(&self) -> Iter<'_, A, D>
    where
        S: Data,
    {
        Iter {
            iter: self.store().iter(),
        }
    }
    /// returns a mutable iterator over the elements of the tensor
    pub fn iter_mut(&mut self) -> IterMut<'_, A, D>
    where
        S: DataMut,
    {
        IterMut {
            iter: self.store_mut().iter_mut(),
        }
    }
}
