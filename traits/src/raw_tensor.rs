/*
    Appellation: tensor <module>
    Contrib: @FL03
*/
use crate::scalar::Scalar;
use ndarray::{ArrayBase, DataMut, DataOwned, Dimension, OwnedRepr, RawData, ShapeBuilder};
use num_traits::{FromPrimitive, One, Signed, Zero};

/// The [`RawTensor`] trait defines the base interface for all tensors,
pub trait RawTensor<S, D, A = <S as RawData>::Elem>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Tensor<_S, _D>
    where
        _S: RawData,
        _D: Dimension;

    private!();

    /// returns the number of dimensions of the object
    fn dim(&self) -> D::Pattern;
    /// returns the shape of the object
    fn raw_dim(&self) -> D;
    /// returns the shape of the object
    fn shape(&self) -> &[usize];
    /// returns the number of elements in the object
    fn size(&self) -> usize {
        self.shape().iter().product()
    }
}

pub trait TensorIter<'a, A: 'a, D: Dimension>: Iterator<Item = &'a A> {}

/// The [`Tensor`] trait extends the [`RawTensor`] trait to provide additional functionality
/// for tensors, such as creating tensors from shapes, applying functions, and iterating over
/// elements. It is generic over the element type `A` and the dimension type `D
pub trait NdTensor<S, D, A = <S as RawData>::Elem>: RawTensor<S, D, A>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// Create a new tensor with the given shape and a function to fill it
    fn from_shape_with_fn<Sh, F>(shape: Sh, f: F) -> Self::Tensor<S, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        F: FnMut(D::Pattern) -> A,
        S: DataOwned,
        Self: Sized;
    /// Create a new tensor with the given shape and value
    fn from_shape_with_value<Sh>(shape: Sh, value: A) -> Self::Tensor<S, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        A: Clone,
        S: DataOwned,
        Self: Sized;
    /// Create a new tensor with the given shape and all values set to their default
    fn default<Sh>(shape: Sh) -> Self::Tensor<S, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        A: Clone + Default,
        S: DataOwned,
        Self: Sized,
    {
        Self::from_shape_with_value(shape, A::default())
    }
    /// create a new tensor with the given shape and all values set to one
    fn ones<Sh>(shape: Sh) -> Self::Tensor<S, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        A: Clone + One,
        S: DataOwned,
        Self: Sized,
    {
        Self::from_shape_with_value(shape, A::one())
    }
    /// create a new tensor with the given shape and all values set to zero
    fn zeros<Sh>(shape: Sh) -> Self::Tensor<S, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        A: Clone + Zero,
        S: DataOwned,
        Self: Sized,
    {
        Self::from_shape_with_value(shape, <A>::zero())
    }
    /// returns a reference to the data of the object
    fn data(&self) -> &Self::Tensor<S, D>;
    /// returns a mutable reference to the data of the object
    fn data_mut(&mut self) -> &mut Self::Tensor<S, D>;
    #[doc(hidden)]
    /// sets the data of the object and returns a mutable reference to the object
    fn set_data(&mut self, data: Self::Tensor<S, D>) -> &mut Self {
        *self.data_mut() = data;
        self
    }

    /// returns a new tensor with the same shape as the object and the given function applied
    fn apply_mut<F>(&mut self, f: F)
    where
        A: Clone,
        S: DataMut,
        F: FnMut(A) -> A;

    /// returns a new tensor with the same shape as the object and the given function applied
    /// to each element
    fn apply<F, B>(&self, f: F) -> Self::Tensor<OwnedRepr<B>, D>
    where
        F: FnMut(A) -> B,
        A: Clone,
        S: DataOwned;

    fn sum(&self) -> A
    where
        A: Clone + core::iter::Sum<A>,
        S: DataOwned;

    fn abs(&self) -> Self::Tensor<OwnedRepr<A>, D>
    where
        A: Clone + Signed,
        S: DataOwned,
    {
        self.apply(|x| x.abs())
    }
    fn mean(&self) -> A
    where
        A: Clone + core::ops::Div<Output = A> + FromPrimitive + core::iter::Sum,
        S: DataOwned,
    {
        let sum = self.sum();
        let count = self.size();
        sum / A::from_usize(count).unwrap()
    }
    fn neg(&self) -> Self::Tensor<OwnedRepr<A>, D>
    where
        A: Clone + core::ops::Neg<Output = A>,
        S: DataOwned,
    {
        self.apply(|x| -x)
    }

    fn pow2(&self) -> Self::Tensor<OwnedRepr<A>, D>
    where
        A: Scalar,
        S: DataOwned,
    {
        let two = A::from_usize(2).unwrap();
        self.apply(|x| x.pow(two))
    }
}

/*
 ************* Implementations *************
*/

impl<A, S, D> RawTensor<S, D, A> for ArrayBase<S, D, A>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
    type Tensor<U: RawData, V: Dimension> = ArrayBase<U, V>;

    seal!();

    fn dim(&self) -> D::Pattern {
        self.dim()
    }

    fn raw_dim(&self) -> D {
        self.raw_dim()
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }
}

impl<A, S, D> NdTensor<S, D, A> for ArrayBase<S, D, A>
where
    S: DataOwned<Elem = A>,
    D: Dimension,
{
    fn from_shape_with_fn<Sh, F>(shape: Sh, f: F) -> Self::Tensor<S, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        F: FnMut(<D as Dimension>::Pattern) -> A,
        S: DataOwned,
        Self: Sized,
    {
        ArrayBase::from_shape_fn(shape, f)
    }
    fn from_shape_with_value<Sh>(shape: Sh, value: A) -> Self::Tensor<S, D>
    where
        A: Clone,
        Sh: ShapeBuilder<Dim = D>,
        S: DataOwned,
        Self: Sized,
    {
        ArrayBase::from_elem(shape, value)
    }
    fn data(&self) -> &Self::Tensor<S, D> {
        self
    }

    fn data_mut(&mut self) -> &mut Self::Tensor<S, D> {
        self
    }

    fn apply_mut<F>(&mut self, f: F)
    where
        A: Clone,
        F: FnMut(A) -> A,
        S: DataMut,
    {
        self.mapv_inplace(f)
    }

    fn apply<F, B>(&self, mut f: F) -> Self::Tensor<OwnedRepr<B>, D>
    where
        A: Clone,
        S: DataOwned,
        F: FnMut(A) -> B,
    {
        self.mapv(|x| f(x))
    }

    fn sum(&self) -> A
    where
        A: Clone + core::iter::Sum,
    {
        self.iter().cloned().sum()
    }
}
