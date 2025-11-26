/*
    Appellation: tensor <module>
    Contrib: @FL03
*/
use crate::scalar::Scalar;
use ndarray::{
    ArrayBase, DataMut, DataOwned, Dimension, OwnedRepr, RawData, ShapeBuilder
};
use num_traits::{One, Signed, Zero};

/// The [`RawTensor`] trait defines the base interface for all tensors,
pub trait RawTensor<A, D> {
    type Repr: RawData<Elem = A>;
    type Container<U: RawData, V: Dimension>;

    private!();

    fn len(&self) -> usize;
}

pub trait TensorIter<'a, A: 'a, D: Dimension>: Iterator<Item = &'a A> {}
/// The [`Tensor`] trait extends the [`RawTensor`] trait to provide additional functionality
/// for tensors, such as creating tensors from shapes, applying functions, and iterating over
/// elements. It is generic over the element type `A` and the dimension type `D
pub trait NdTensor<A, D>: RawTensor<A, D>
where
    D: Dimension,
{
    /// Create a new tensor with the given shape and a function to fill it
    fn from_shape_with_fn<Sh, F>(shape: Sh, f: F) -> Self::Container<Self::Repr, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        F: FnMut(D::Pattern) -> A,
        Self: Sized;
    /// Create a new tensor with the given shape and value
    fn from_shape_with_value<Sh>(shape: Sh, value: A) -> Self::Container<Self::Repr, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        Self: Sized;
    /// Create a new tensor with the given shape and all values set to their default
    fn default<Sh>(shape: Sh) -> Self::Container<Self::Repr, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        Self: Sized,
        A: Default,
    {
        Self::from_shape_with_value(shape, A::default())
    }
    /// create a new tensor with the given shape and all values set to one
    fn ones<Sh>(shape: Sh) -> Self::Container<Self::Repr, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        Self: Sized,
        A: Clone + One,
    {
        Self::from_shape_with_value(shape, A::one())
    }
    /// create a new tensor with the given shape and all values set to zero
    fn zeros<Sh>(shape: Sh) -> Self::Container<Self::Repr, D>
    where
        Sh: ShapeBuilder<Dim = D>,
        Self: Sized,
        A: Clone + Zero,
    {
        Self::from_shape_with_value(shape, <A>::zero())
    }
    /// returns a reference to the data of the object
    fn data(&self) -> &Self::Container<Self::Repr, D>;
    /// returns a mutable reference to the data of the object
    fn data_mut(&mut self) -> &mut Self::Container<Self::Repr, D>;
    /// returns the number of dimensions of the object
    fn dim(&self) -> D::Pattern;
    /// returns the shape of the object
    fn raw_dim(&self) -> D;
    /// returns the shape of the object
    fn shape(&self) -> &[usize];
    /// returns a new tensor with the same shape as the object and the given function applied
    /// to each element
    fn apply<F, B>(&self, f: F) -> Self::Container<OwnedRepr<B>, D>
    where
        F: FnMut(A) -> B;
    /// returns a new tensor with the same shape as the object and the given function applied
    fn apply_mut<F>(&mut self, f: F)
    where
        Self::Repr: DataMut,
        F: FnMut(A) -> A;

    fn mean(&self) -> A
    where
        A: Scalar,
    {
        let sum = self.sum();
        let count = self.len();
        sum / A::from_usize(count).unwrap()
    }
    #[doc(hidden)]
    /// sets the data of the object and returns a mutable reference to the object
    fn set_data(&mut self, data: Self::Container<Self::Repr, D>) -> &mut Self {
        *self.data_mut() = data;
        self
    }

    fn sum(&self) -> A
    where
        A: Clone + core::iter::Sum;

    fn pow2(&self) -> Self::Container<OwnedRepr<A>, D>
    where
        A: Scalar,
    {
        let two = A::from_usize(2).unwrap();
        self.apply(|x| x.pow(two))
    }

    fn abs(&self) -> Self::Container<OwnedRepr<A>, D>
    where
        A: Signed,
    {
        self.apply(|x| x.abs())
    }

    fn neg(&self) -> Self::Container<OwnedRepr<A>, D>
    where
        A: core::ops::Neg<Output = A>,
    {
        self.apply(|x| -x)
    }
}

/*
 ************* Implementations *************
*/

impl<A, S, D> RawTensor<A, D> for ArrayBase<S, D>
where
    S: RawData<Elem = A>,
    D: Dimension,
{
    type Repr = S;
    type Container<U: RawData, V: Dimension> = ArrayBase<U, V>;

    seal!();

    fn len(&self) -> usize {
        self.len()
    }
}

impl<A, S, D> NdTensor<A, D> for ArrayBase<S, D>
where
    S: DataOwned<Elem = A>,
    A: Scalar,
    D: Dimension,
{
    fn from_shape_with_value<Sh>(shape: Sh, value: A) -> Self::Container<Self::Repr, D>
    where
        Self: Sized,
        Sh: ndarray::ShapeBuilder<Dim = D>,
    {
        Self::Container::<S, D>::from_elem(shape, value)
    }

    fn from_shape_with_fn<Sh, F>(shape: Sh, f: F) -> Self::Container<Self::Repr, D>
    where
        Self: Sized,
        Sh: ShapeBuilder<Dim = D>,
        F: FnMut(D::Pattern) -> A,
    {
        Self::Container::<S, D>::from_shape_fn(shape, f)
    }

    fn data(&self) -> &Self::Container<Self::Repr, D> {
        self
    }

    fn data_mut(&mut self) -> &mut Self::Container<Self::Repr, D> {
        self
    }

    fn dim(&self) -> D::Pattern {
        self.dim()
    }

    fn raw_dim(&self) -> D {
        self.raw_dim()
    }

    fn shape(&self) -> &[usize] {
        self.shape()
    }

    fn sum(&self) -> A
    where
        A: Clone + core::iter::Sum,
    {
        self.iter().cloned().sum()
    }

    fn apply<F, B>(&self, f: F) -> Self::Container<OwnedRepr<B>, D>
    where
        F: FnMut(A) -> B,
    {
        self.mapv(f)
    }

    fn apply_mut<F>(&mut self, f: F)
    where
        F: FnMut(A) -> A,
        S: DataMut,
    {
        self.mapv_inplace(f)
    }
}

