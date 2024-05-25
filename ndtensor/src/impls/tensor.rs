/*
    Appellation: tensor <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{TensorBase, TensorMode};
use core::borrow::{Borrow, BorrowMut};
use core::ops::{Deref, DerefMut, Index, IndexMut};
use nd::{ArrayBase, Data, DataMut, Dimension, NdIndex, RawData, RawDataClone};

impl<A, S, D, K> AsRef<ArrayBase<S, D>> for TensorBase<S, D, K>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn as_ref(&self) -> &ArrayBase<S, D> {
        &self.data
    }
}

impl<A, S, D, K> AsMut<ArrayBase<S, D>> for TensorBase<S, D, K>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn as_mut(&mut self) -> &mut ArrayBase<S, D> {
        &mut self.data
    }
}

impl<A, S, D, K> Borrow<ArrayBase<S, D>> for TensorBase<S, D, K>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn borrow(&self) -> &ArrayBase<S, D> {
        &self.data
    }
}

impl<A, S, D, K> BorrowMut<ArrayBase<S, D>> for TensorBase<S, D, K>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn borrow_mut(&mut self) -> &mut ArrayBase<S, D> {
        &mut self.data
    }
}

impl<S, D, K> Clone for TensorBase<S, D, K>
where
    D: Dimension,
    K: Clone,
    S: RawDataClone,
{
    fn clone(&self) -> Self {
        TensorBase {
            id: self.id,
            ctx: self.ctx.clone(),
            data: self.data.clone(),
        }
    }
}

impl<S, D, K> Copy for TensorBase<S, D, K>
where
    D: Copy + Dimension,
    K: Copy,
    S: Copy + RawDataClone,
{
}

impl<A, S, D, K> Deref for TensorBase<S, D, K>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    type Target = ArrayBase<S, D>;

    fn deref(&self) -> &Self::Target {
        &self.data
    }
}

impl<A, S, D, K> DerefMut for TensorBase<S, D, K>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.data
    }
}

impl<S, D, K, I> Index<I> for TensorBase<S, D, K>
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

impl<S, D, K, I> IndexMut<I> for TensorBase<S, D, K>
where
    D: Dimension,
    I: NdIndex<D>,
    S: DataMut,
{
    fn index_mut(&mut self, index: I) -> &mut Self::Output {
        &mut self.data[index]
    }
}

impl<A, S, D, K> PartialEq for TensorBase<S, D, K>
where
    A: PartialEq,
    D: Dimension,
    S: Data<Elem = A>,
{
    fn eq(&self, other: &Self) -> bool {
        self.data == other.data
    }
}

impl<S, D, K, U> PartialEq<U> for TensorBase<S, D, K>
where
    D: Dimension,
    S: Data,
    ArrayBase<S, D>: PartialEq<U>,
{
    fn eq(&self, other: &U) -> bool {
        self.data.eq(other)
    }
}

macro_rules! impl_fmt {
    ($($trait:ident($($fmt:tt)*)),*) => {
        $(
            impl_fmt!(@impl $trait($($fmt)*));
        )*
    };
    (@impl $trait:ident($($fmt:tt)*)) => {
        impl<A, S, D, K> ::core::fmt::$trait for TensorBase<S, D, K>
        where
            A: ::core::fmt::$trait,
            D: Dimension,
            K: $crate::TensorMode,
            S: Data<Elem = A>,
        {
            fn fmt(&self, f: &mut core::fmt::Formatter) -> ::core::fmt::Result {
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


impl<S, D, K> From<ArrayBase<S, D>> for TensorBase<S, D, K>
where
    D: Dimension,
    K: TensorMode,
    S: RawData,
{
    fn from(data: ArrayBase<S, D>) -> Self {
        TensorBase::new(data)
    }
}