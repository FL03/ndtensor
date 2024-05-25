/*
    Appellation: tensor <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::{Normal, TensorError, TensorId, TensorMode, Variable};
use crate::Context;
use nd::*;

/// This is the base tensor object, providing additional functionality to the wrapped [ArrayBase](ndarray::ArrayBase).
///
/// ### `context`
/// 
/// The context of the tensor provides additional information about the object, including the tensor's _kind_, _rank_, and more.
/// For simplicity, the [TensorBase] relies upon the [Context] to store the type parameter `K` which represents the tensor's _kind_.
/// The _kind_ of the tensor is used to determine the tensor's behavior and how it interacts with the computational graph*.
pub struct TensorBase<S, D = IxDyn, K = Normal>
where
    D: Dimension,
    S: RawData,
{
    pub(crate) id: TensorId,
    pub(crate) ctx: Context<K>,
    pub(crate) data: ArrayBase<S, D>,
}

impl<A, S, D, K> TensorBase<S, D, K>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// Returns a slice of the tensor.
    pub fn as_slice(&self) -> Option<&[A]>
    where
        S: Data,
    {
        self.data().as_slice()
    }
    /// Returns a mutable slice of the tensor.
    pub fn as_mut_slice(&mut self) -> Option<&mut [A]>
    where
        S: DataMut,
    {
        self.data_mut().as_slice_mut()
    }
    /// Performs an elementwise assignment of the values in the tensor.
    pub fn assign<T, E>(&mut self, value: &ArrayBase<T, E>)
    where
        A: Clone,
        E: Dimension,
        S: DataMut,
        T: Data<Elem = A>,
    {
        self.data_mut().assign(value);
    }
    /// Returns a immutable iterator over the axes of the tensor.
    /// See [Axes](ndarray::iter::Axes) for more details.
    pub fn axes(&self) -> iter::Axes<'_, D> {
        self.data().axes()
    }

    pub fn boxed(self) -> Box<Self> {
        Box::new(self)
    }
    /// Get an immutable reference to the [context](Context) of the tensor.
    pub const fn ctx(&self) -> &Context<K> {
        &self.ctx
    }
    /// Get a mutable reference to the [context](Context) of the tensor.
    pub fn ctx_mut(&mut self) -> &mut Context<K> {
        &mut self.ctx
    }
    /// Returns a reference to the underlying [ArrayBase]
    pub const fn data(&self) -> &ArrayBase<S, D> {
        &self.data
    }
    /// Returns a mutable reference to the underlying [ArrayBase]
    pub fn data_mut(&mut self) -> &mut ArrayBase<S, D> {
        &mut self.data
    }

    pub fn detach(&self) -> crate::TensorView<'_, A, D, K>
    where
        K: Copy,
        S: Data,
    {
        TensorBase {
            id: self.id,
            ctx: self.ctx,
            data: self.data.view(),
        }
    }
    /// Returns the dimension of the tensor.
    pub fn dim(&self) -> D::Pattern {
        self.data().dim()
    }
    /// Returns the unique identifier of the tensor.
    pub const fn id(&self) -> &TensorId {
        &self.id
    }
    /// Returns true tensor is of Rank(0); i.e., a scalar.
    pub fn is_scalar(&self) -> bool {
        self.ndim() == 0
    }
    /// Returns true if the tensor is a variable tensor.
    pub fn is_variable(&self) -> bool where K: 'static {
        Variable::is::<K>()
    }
    /// Returns a immutable iterator over the elements in the tensor.
    /// See [Iter](ndarray::iter::Iter) for more details.
    pub fn iter(&self) -> iter::Iter<'_, A, D>
    where
        S: Data,
    {
        self.data().iter()
    }
    /// Returns a mutable iterator over the elements in the tensor.
    /// See [IterMut](ndarray::iter::IterMut) for more details.
    pub fn iter_mut(&mut self) -> iter::IterMut<'_, A, D>
    where
        S: DataMut,
    {
        self.data.iter_mut()
    }
    /// Returns the length of the tensor; i.e., the number of elements in the tensor.
    pub fn len(&self) -> usize {
        self.data().len()
    }

    pub fn map<'a, F, B>(&'a self, f: F) -> crate::Tensor<B, D, K>
    where
        A: 'a,
        F: FnMut(&'a A) -> B,
        K: TensorMode,
        S: Data,
    {
        self.data().map(f).into()
    }
    /// Applies a closure to each element of the tensor in place.
    pub fn map_inplace<'a, F, B>(&'a mut self, f: F)
    where
        A: 'a,
        F: FnMut(&'a mut A),
        S: DataMut,
    {
        self.data_mut().map_inplace(f);
    }
    /// Applies a closure to each element of the tensor and returns a new tensor with the results.
    pub fn mapv<B, F>(&mut self, f: F) -> crate::Tensor<B, D, K>
    where
        A: Clone,
        F: FnMut(A) -> B,
        K: TensorMode,
        S: DataMut,
    {
        self.data().mapv(f).into()
    }
    /// Returns the number of dimensions of the tensor.
    pub fn ndim(&self) -> usize {
        self.data().ndim()
    }
    /// Returns a reference to the [Dimension] of the tensor
    pub fn raw_dim(&self) -> D {
        self.data().raw_dim()
    }
    /// Returns a reference to the shape of the tensor.
    pub fn shape(&self) -> &[usize] {
        self.data().shape()
    }
    /// Returns a reference to the stride of the tensor
    pub fn strides(&self) -> &[isize] {
        self.data().strides()
    }

    pub fn with_ctx<J>(self, ctx: Context<J>) -> TensorBase<S, D, J>
    where
        J: TensorMode,
    {
        TensorBase {
            id: self.id,
            ctx,
            data: self.data,
        }
    }

    pub fn with_data(self, data: ArrayBase<S, D>) -> Self {
        TensorBase { data, ..self }
    }

    apply_view!(cell_view(&mut self) -> crate::TensorView<'_, MathCell<A>, D, K> where S: DataMut);

    apply_view!(into_owned(self) -> crate::Tensor<A, D, K> where A: Clone, S: DataOwned);

    apply_view!(to_owned(&self) -> crate::Tensor<A, D, K> where A: Clone, S: Data);

    apply_view!(into_shared(self) -> crate::ArcTensor<A, D, K> where S: DataOwned);

    apply_view!(to_shared(&self) -> crate::ArcTensor<A, D, K> where A: Clone, S: Data);

    apply_view!(raw_view(&self) -> crate::RawTensorView<A, D, K> where S: RawData);

    apply_view!(raw_view_mut(&mut self) -> crate::RawTensorViewMut<A, D, K> where S: RawDataMut);

    apply_view!(view(&self) -> crate::TensorView<'_, A, D, K> where S: Data);

    apply_view!(view_mut(&mut self) -> crate::TensorViewMut<'_, A, D, K> where S: DataMut);
}


impl<A, S, D, K> TensorBase<S, D, K>
where
    D: Dimension,
    K: TensorMode,
    S: RawData<Elem = A>,
{
    pub(crate) fn new(data: ArrayBase<S, D>) -> Self {
        TensorBase {
            id: TensorId::new(),
            ctx: Context::<K>::from_arr(&data),
            data,
        }
    }

    pub fn from_arr(data: ArrayBase<S, D>) -> Self {
        Self::new(data)
    }

    pub fn try_from_arr<D2>(data: ArrayBase<S, D2>) -> Result<Self, TensorError>
    where
        D2: Dimension,
    {
        let tensor = Self::from_arr(data.into_dimensionality::<D>()?);
        Ok(tensor)
    }
    /// Returns 1D array containing the diagonal elements.
    pub fn diag(&self) -> crate::TensorView<'_, A, Ix1, K>
    where
        S: Data,
    {
        self.data().diag().into()
    }
    /// Returns a mutable 1D array containing the diagonal elements.
    pub fn diag_mut(&mut self) -> crate::TensorViewMut<'_, A, Ix1, K>
    where
        S: DataMut,
    {
        self.data.diag_mut().into()
    }

    pub fn into_dimensionality<D2>(self) -> Result<TensorBase<S, D2, K>, TensorError>
    where
        D2: Dimension,
    {
        let data = self.data.into_dimensionality::<D2>()?;
        Ok(TensorBase {
            id: self.id,
            ctx: self.ctx,
            data,
        })
    }

    pub fn into_dyn(self) -> TensorBase<S, IxDyn, K> {
        TensorBase {
            id: self.id,
            ctx: self.ctx,
            data: self.data.into_dyn(),
        }
    }

    pub fn to_dyn(&self) -> TensorBase<S, IxDyn, K>
    where
        S: RawDataClone,
    {
        TensorBase {
            id: self.id,
            ctx: self.ctx,
            data: self.data().clone().into_dyn(),
        }
    }

    pub fn slice<I>(&self, info: I) -> crate::TensorView<'_, A, I::OutDim, K>
    where
        I: SliceArg<D>,
        S: Data,
    {
        let data = self.data().slice(info);
        TensorBase {
            id: self.id,
            ctx: self.ctx,
            data,
        }
    }

    pub fn slice_mut<I>(&mut self, info: I) -> crate::TensorViewMut<'_, A, I::OutDim, K>
    where
        I: SliceArg<D>,
        S: DataMut,
    {
        let data = self.data.slice_mut(info);
        TensorBase {
            id: self.id,
            ctx: self.ctx,
            data,
        }
    }

    pub fn into_variable(self) -> TensorBase<S, D, Variable> {
        TensorBase {
            id: self.id,
            ctx: self.ctx.into_variable(),
            data: self.data,
        }
    }
}

impl<A, S, D> TensorBase<S, D, Normal>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub fn normal() -> Self
    where
        A: Default,
        S: DataOwned,
    {
        Self::from_arr(Default::default())
    }
}

impl<A, S, D> TensorBase<S, D, Variable>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    pub fn variable() -> Self
    where
        A: Default,
        S: DataOwned,
    {
        Self::from_arr(Default::default())
    }
}
