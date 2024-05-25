/*
    Appellation: linalg <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{Tensor, TensorBase, TensorMode};
use nd::{Data, LinalgScalar};
use nd::linalg::Dot;
use nd::prelude::*;

impl<A, B, C, S, T, D, E, F, K> Dot<TensorBase<T, E, K>> for TensorBase<S, D, K>
where
    A: LinalgScalar,
    D: Dimension,
    E: Dimension,
    F: Dimension,
    K: TensorMode,
    S: Data<Elem = A>,
    T: Data<Elem = B>,
    ArrayBase<S, D>: Dot<ArrayBase<T, E>, Output = Array<C, F>>,
{
    type Output = Tensor<C, F, K>;

    fn dot(&self, rhs: &TensorBase<T, E, K>) -> Self::Output {
        self.data().dot(rhs.data()).into()
    }
}

impl<A, B, C, S, T, D, E, F, K> Dot<ArrayBase<T, E>> for TensorBase<S, D, K>
where
    A: LinalgScalar,
    D: Dimension,
    E: Dimension,
    F: Dimension,
    K: TensorMode,
    S: Data<Elem = A>,
    T: Data<Elem = B>,
    ArrayBase<S, D>: Dot<ArrayBase<T, E>, Output = Array<C, F>>,
{
    type Output = Tensor<C, F, K>;

    fn dot(&self, rhs: &ArrayBase<T, E>) -> Self::Output {
        self.data().dot(rhs).into()
    }
}