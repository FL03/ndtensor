/*
    Appellation: create <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::{TensorError, TensorMode};
use crate::TensorBase;
use core::ops::Mul;
use nd::*;
use num::{Num, NumCast, One, Zero};

impl<A, S, K> TensorBase<S, Ix0, K>
where
    K: TensorMode,
    S: Data<Elem = A>,
{
    pub fn from_scalar(scalar: A) -> Self
    where
        A: Clone,
        S: DataOwned,
    {
        ArrayBase::from_elem((), scalar).into()
    }
}

impl<A, S, K> TensorBase<S, IxDyn, K>
where
    K: TensorMode,
    S: RawData<Elem = A>,
{
    pub fn ndtensor<D>(data: ArrayBase<S, D>) -> TensorBase<S, IxDyn, K>
    where
        D: Dimension,
    {
        data.into_dyn().into()
    }
}

impl<A, S, D, K> TensorBase<S, D, K>
where
    D: Dimension,
    K: TensorMode,
    S: DataOwned<Elem = A>,
{
    pub fn default_like(&self) -> Self
    where
        A: Clone + Default,
    {
        Self::default(self.dim())
    }

    pub fn fill(shape: D, elem: A) -> Self
    where
        A: Clone,
    {
        Self::from_elem(shape, elem)
    }

    pub fn linshape(shape: impl IntoDimension<Dim = D>) -> Result<TensorBase<S, D, K>, ShapeError>
    where
        A: Clone + num::Float,
    {
        let dim = shape.into_dimension();
        let n = {
            let tmp = dim.as_array_view();
            tmp.product()
        };
        TensorBase::<S, ndarray::Ix1, K>::linspace(A::zero(), A::from(n).unwrap() - A::one(), n)
            .into_shape(dim)
    }

    pub fn ones_like(&self) -> Self
    where
        A: Clone + One,
    {
        Self::ones(self.dim())
    }

    pub fn zeros_like(&self) -> Self
    where
        A: Clone + Zero,
    {
        Self::zeros(self.dim())
    }

    ndcreate!(default<Sh>(shape: Sh) -> Self where A: Clone + Default, Sh: ShapeBuilder<Dim = D>);

    ndcreate!(from_elem(shape: D, elem: A) -> Self where A: Clone);

    ndcreate!(from_shape_vec(shape: D, data: Vec<S::Elem>) -> Result<Self, TensorError> where S: DataOwned);

    ndcreate!(ones<Sh>(shape: Sh) -> Self where A: Clone + One, Sh: ShapeBuilder<Dim = D>);

    ndcreate!(zeros<Sh>(shape: Sh) -> Self where A: Clone + Zero, Sh: ShapeBuilder<Dim = D>);
}

impl<A, S, K> TensorBase<S, Ix1, K>
where
    K: TensorMode,
    S: DataOwned<Elem = A>,
{
    pub fn arange(start: A, end: A, step: A) -> Self
    where
        A: Clone + num::Float,
    {
        Self::range(start, end, step)
    }

    ndcreate!(linspace(start: A, end: A, num: usize) -> Self where A: num::Float);

    ndcreate!(logspace(base: A, start: A, end: A, num: usize) -> Self where A: num::Float);

    ndcreate!(range(start: A, end: A, step: A) -> Self where A: Clone + num::Float);
}

impl<A, S, K> TensorBase<S, Ix2, K>
where
    K: TensorMode,
    S: DataOwned<Elem = A>,
{
    ndcreate!(eye(n: usize) -> Self where A: Clone + Zero + One, S: DataMut);

    pub fn identity(n: usize) -> Self
    where
        A: Clone + Zero + One,
        S: DataMut,
    {
        Self::eye(n)
    }
}

impl<A> One for crate::Tensor<A, Ix0, crate::Variable>
where
    A: Clone + NumCast + One + Mul<Output = A>,
{
    fn one() -> Self {
        Self::from_scalar(A::one())
    }
}

impl<A> Zero for crate::Tensor<A, Ix0, crate::Variable>
where
    A: Clone + Zero + NumCast,
{
    fn zero() -> Self {
        Self::from_scalar(A::zero())
    }

    fn is_zero(&self) -> bool {
        self.data.iter().all(|x| x.is_zero())
    }
}

impl<A> Num for crate::Tensor<A, Ix0, crate::Variable>
where
    A: Clone + Num + NumCast,
{
    type FromStrRadixErr = A::FromStrRadixErr;

    fn from_str_radix(str: &str, radix: u32) -> Result<Self, Self::FromStrRadixErr> {
        A::from_str_radix(str, radix).map(Self::from_scalar)
    }
}
