/*
    Appellation: tensor <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::{Normal, TensorError, TensorExpr, TensorId, TensorMode, TensorOp, Variable};
use crate::Context;
use core::borrow::{Borrow, BorrowMut};
use core::marker::PhantomData;
use nd::*;

/// This is the base tensor object, providing additional functionality to the wrapped [ArrayBase](ndarray::ArrayBase).
///
///
pub struct TensorBase<S, D = IxDyn, K = Normal>
where
    D: Dimension,
    S: RawData,
{
    pub(crate) id: TensorId,
    pub(crate) ctx: Context,
    pub(crate) data: ArrayBase<S, D>,
    pub(crate) op: TensorOp<S>,
    pub(crate) _kind: PhantomData<K>,
}

impl<A, S, D, K> TensorBase<S, D, K>
where
    D: Dimension,
    K: TensorMode,
    S: RawData<Elem = A>,
{
    pub(crate) fn new(data: ArrayBase<S, D>, op: Option<TensorExpr<S>>, kind: bool) -> Self {
        create!(data, kind: kind, op: op.into())
    }

    pub fn from_arr(data: ArrayBase<S, D>) -> Self {
        create!(data,)
    }

    pub fn try_from_arr<D2>(data: ArrayBase<S, D2>) -> Result<Self, TensorError>
    where
        D2: Dimension,
    {
        let tensor = Self::from_arr(data.into_dimensionality::<D>()?);
        Ok(tensor)
    }

    pub fn as_slice(&self) -> Option<&[A]>
    where
        S: Data,
    {
        self.data().as_slice()
    }

    pub fn as_mut_slice(&mut self) -> Option<&mut [A]>
    where
        S: DataMut,
    {
        self.data_mut().as_slice_mut()
    }

    pub fn assign<T, E>(&mut self, value: &ArrayBase<T, E>)
    where
        A: Clone,
        E: Dimension,
        S: DataMut,
        T: Data<Elem = A>,
    {
        self.data_mut().assign(value);
    }

    pub fn axes(&self) -> iter::Axes<'_, D> {
        self.data().axes()
    }

    pub fn boxed(self) -> Box<Self> {
        Box::new(self)
    }
    /// Get an immutable reference to the [context](Context) of the tensor.
    pub const fn ctx(&self) -> &Context {
        &self.ctx
    }
    /// Get a mutable reference to the [context](Context) of the tensor.
    pub fn ctx_mut(&mut self) -> &mut Context {
        &mut self.ctx
    }

    pub const fn data(&self) -> &ArrayBase<S, D> {
        &self.data
    }

    pub fn data_mut(&mut self) -> &mut ArrayBase<S, D> {
        &mut self.data
    }

    pub fn detach(&self) -> crate::TensorView<'_, A, D, K>
    where
        S: Data,
    {
        TensorBase {
            id: self.id,
            ctx: self.ctx,
            data: self.data.view(),
            op: TensorOp::none(),
            _kind: PhantomData::<K>,
        }
    }

    pub fn diag(&self) -> crate::TensorView<'_, A, Ix1, K>
    where
        S: Data,
    {
        TensorBase::new(self.data().diag(), None, false)
    }

    pub fn diag_mut(&mut self) -> crate::TensorViewMut<'_, A, Ix1, K>
    where
        S: DataMut,
    {
        TensorBase::new(self.data.diag_mut(), None, false)
    }

    pub fn dim(&self) -> D::Pattern {
        self.data.dim()
    }

    /// Returns the unique identifier of the tensor.
    pub const fn id(&self) -> &TensorId {
        &self.id
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
            op: self.op,
            _kind: PhantomData::<K>,
        })
    }

    pub fn into_dyn(self) -> TensorBase<S, IxDyn, K> {
        TensorBase {
            id: self.id,
            ctx: self.ctx,
            data: self.data.into_dyn(),
            op: self.op,
            _kind: PhantomData::<K>,
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
            op: self.op.clone(),
            _kind: PhantomData::<K>,
        }
    }

    pub fn is_scalar(&self) -> bool {
        self.ndim() == 0
    }

    pub fn is_variable(&self) -> bool {
        self.ctx().is_variable()
    }

    pub fn iter(&self) -> iter::Iter<'_, A, D>
    where
        S: Data,
    {
        self.data().iter()
    }

    pub fn iter_mut(&mut self) -> iter::IterMut<'_, A, D>
    where
        S: DataMut,
    {
        self.data.iter_mut()
    }

    pub fn len(&self) -> usize {
        self.data().len()
    }

    pub fn ndim(&self) -> usize {
        self.data().ndim()
    }

    pub fn raw_dim(&self) -> D {
        self.data().raw_dim()
    }

    pub fn shape(&self) -> &[usize] {
        self.data().shape()
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
            op: self.op.view(),
            _kind: PhantomData::<K>,
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
            op: self.op.view_mut(),
            _kind: PhantomData::<K>,
        }
    }

    pub fn strides(&self) -> &[isize] {
        self.data().strides()
    }

    /// Gets an immutable reference to the operations of the tensor.
    pub fn op(&self) -> Option<&TensorExpr<S>> {
        self.op.as_ref()
    }

    pub fn into_variable(self) -> TensorBase<S, D, Variable> {
        TensorBase {
            id: self.id,
            ctx: self.ctx,
            data: self.data,
            op: self.op,
            _kind: PhantomData::<Variable>,
        }
    }

    pub fn with_ctx(self, ctx: Context) -> Self {
        TensorBase { ctx, ..self }
    }

    pub fn with_data(self, data: ArrayBase<S, D>) -> Self {
        TensorBase { data, ..self }
    }

    pub fn with_op(self, op: impl Into<TensorOp<S>>) -> Self {
        TensorBase {
            op: op.into(),
            ..self
        }
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

impl<S, D, K> Borrow<ArrayBase<S, D>> for TensorBase<S, D, K>
where
    D: Dimension,
    S: RawData,
{
    fn borrow(&self) -> &ArrayBase<S, D> {
        &self.data
    }
}

impl<S, D, K> BorrowMut<ArrayBase<S, D>> for TensorBase<S, D, K>
where
    D: Dimension,
    S: RawData,
{
    fn borrow_mut(&mut self) -> &mut ArrayBase<S, D> {
        &mut self.data
    }
}

impl<S, D, K> Clone for TensorBase<S, D, K>
where
    D: Dimension,
    S: RawDataClone,
{
    fn clone(&self) -> Self {
        TensorBase {
            id: self.id,
            ctx: self.ctx.clone(),
            data: self.data.clone(),
            op: self.op.clone(),
            _kind: PhantomData::<K>,
        }
    }
}

impl<A, S, D, K> PartialEq for TensorBase<S, D, K>
where
    D: Dimension,
    S: Data<Elem = A>,
    A: PartialEq,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<S, D, K, U> PartialEq<U> for TensorBase<S, D, K>
where
    D: Dimension,
    K: TensorMode,
    S: Data,
    ArrayBase<S, D>: PartialEq<U>,
{
    fn eq(&self, other: &U) -> bool {
        self.data().eq(other)
    }
}

impl<S, D, K, I> core::ops::Index<I> for TensorBase<S, D, K>
where
    D: Dimension,
    I: NdIndex<D>,
    S: Data,
{
    type Output = <S as RawData>::Elem;

    fn index(&self, index: I) -> &Self::Output {
        &self.data[index]
    }
}

impl<S, D, K, I> core::ops::IndexMut<I> for TensorBase<S, D, K>
where
    D: Dimension,
    I: NdIndex<D>,
    S: DataMut,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.data[index]
    }
}

macro_rules! impl_fmt {
    ($($trait:ident($($fmt:tt)*)),*) => {
        $(
            impl_fmt!(@impl $trait($($fmt)*));
        )*
    };
    (@impl $trait:ident($($fmt:tt)*)) => {
        impl<A, S, D, K> core::fmt::$trait for TensorBase<S, D, K>
        where
            A: core::fmt::$trait,
            D: Dimension,
            K: $crate::types::TensorMode,
            S: Data<Elem = A>,
        {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> core::fmt::Result {
                write!(f, $($fmt)*, self.data())
            }
        }
    };
}

impl_fmt!(
    Binary("{:b}"),
    Debug("{:?}"),
    Display("{}"),
    LowerExp("{:e}"),
    LowerHex("{:x}")
);
