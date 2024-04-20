/*
    Appellation: ops <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use super::TensorExpr;
use core::borrow::{Borrow, BorrowMut};
use core::ops::{Deref, DerefMut};
use nd::*;

pub struct TensorOp<S1, S2 = S1>(pub(crate) Option<TensorExpr<S1, S2>>)
where
    S1: RawData,
    S2: RawData;


macro_rules! opviews {
    ($call:ident(self) -> $view:ident where $($arg:ident:$($bound:ident)*),*) => {
        pub fn $call(self) -> TensorOp<$view<A>, $view<B>> where $($arg: $($bound)*),* {
            TensorOp(self.0.map(|expr| expr.$call()))
        }
    };
    ($call:ident(self) -> $view:ident) => {
        pub fn $call(self) -> TensorOp<$view<A>, $view<B>> {
            TensorOp(self.0.map(|expr| expr.$call()))
        }
    };
    ($call:ident(&self) -> $view:ident<*$ptr:ident> where $($arg:ident:$($bound:ident)*),*) => {
        pub fn $call(&self) -> TensorOp<$view<*$ptr A>, $view<*$ptr B>> where $($arg: $($bound)*),* {
            TensorOp(self.as_ref().map(|expr| expr.$call()))
        }
    };
    ($call:ident(&self) -> $view:ident<&'_> where $($arg:ident:$($bound:ident)*),*) => {
        pub fn $call(&self) -> TensorOp<$view<&'_ A>, $view<&'_ B>> where $($arg: $($bound)*),* {
            TensorOp(self.as_ref().map(|expr| expr.$call()))
        }
    };
    ($call:ident(&self) -> $view:ident where $($arg:ident:$($bound:ident)*),*) => {
        pub fn $call(&self) -> TensorOp<$view<A>, $view<B>> where $($arg: $($bound)*),* {
            TensorOp(self.as_ref().map(|expr| expr.$call()))
        }
    };
    ($call:ident(&self) -> $view:ident) => {
        pub fn $call(&self) -> TensorOp<$view<A>, $view<B>> {
            TensorOp(self.as_ref().map(|expr| expr.$call()))
        }
    };
    ($call:ident(&mut self) -> $view:ident<*$ptr:ident> where $($arg:ident:$($bound:ident)*),*) => {
        pub fn $call(&mut self) -> TensorOp<$view<*$ptr A>, $view<*$ptr B>> where $($arg: $($bound)*),* {
            TensorOp(self.as_mut().map(|expr| expr.$call()))
        }
    };
    ($call:ident(&mut self) -> $view:ident<&'_ mut> where $($arg:ident:$($bound:ident)*),*) => {
        pub fn $call(&mut self) -> TensorOp<$view<&'_ mut A>, $view<&'_ mut B>> where $($arg: $($bound)*),* {
            TensorOp(self.as_mut().map(|expr| expr.$call()))
        }
    };
    (&mut $call:ident<$view:ident> where $($arg:ident:$($bound:ident)*),*) => {
        pub fn $call(&mut self) -> TensorOp<$view<A>, $view<B>> where $($arg: $($bound)*),* {
            TensorOp(self.as_mut().map(|expr| expr.$call()))
        }
    };
    
    (&mut $call:ident<$view:ident>) => {
        pub fn $call(&mut self) -> TensorOp<$view<A>, $view<B>> {
            TensorOp(self.as_mut().map(|expr| expr.$call()))
        }
    };
}

impl<A, B, S1, S2> TensorOp<S1, S2>
where
    S1: RawData<Elem = A>,
    S2: RawData<Elem = B>,
{
    pub fn new(expr: Option<TensorExpr<S1, S2>>) -> Self {
        TensorOp(expr)
    }

    pub fn none() -> Self {
        TensorOp(None)
    }

    pub fn as_ref(&self) -> Option<&TensorExpr<S1, S2>> {
        self.0.as_ref()
    }

    pub fn as_mut(&mut self) -> Option<&mut TensorExpr<S1, S2>> {
        self.0.as_mut()
    }



    pub fn is_none(&self) -> bool {
        self.0.is_none()
    }

    pub fn is_some(&self) -> bool {
        self.0.is_some()
    }

    pub fn numcast<C>(&self) -> TensorOp<OwnedRepr<C>, OwnedRepr<C>>
    where
        A: Clone + num::ToPrimitive,
        B: Clone + num::ToPrimitive,
        C: num::NumCast,
        S1: Data,
        S2: Data,
    {
        TensorOp(self.as_ref().map(|expr| expr.numcast::<C>()))
    }

    opviews!(into_owned(self) -> OwnedRepr where A: Clone, B: Clone, S1: DataOwned, S2: DataOwned);
    opviews!(to_owned(&self) -> OwnedRepr where A: Clone, B: Clone, S1: Data, S2: Data);
    opviews!(into_shared(self) -> OwnedArcRepr where S1: DataOwned, S2: DataOwned);
    opviews!(to_shared(&self) -> OwnedArcRepr where A: Clone, B: Clone, S1: Data, S2: Data);
    opviews!(raw_view(&self) -> RawViewRepr<*const> where S1: RawData, S2: RawData);
    opviews!(raw_view_mut(&mut self) -> RawViewRepr<*mut> where S1: RawDataMut, S2: RawDataMut);
    opviews!(view(&self) -> ViewRepr<&'_> where S1: Data, S2: Data);
    opviews!(view_mut(&mut self) -> ViewRepr<&'_ mut> where S1: DataMut, S2: DataMut);
}

impl<S1, S2> Borrow<Option<TensorExpr<S1, S2>>> for TensorOp<S1, S2>
where
    S1: RawData,
    S2: RawData,
{
    fn borrow(&self) -> &Option<TensorExpr<S1, S2>> {
        &self.0
    }
}

impl<S1, S2> BorrowMut<Option<TensorExpr<S1, S2>>> for TensorOp<S1, S2>
where
    S1: RawData,
    S2: RawData,
{
    fn borrow_mut(&mut self) -> &mut Option<TensorExpr<S1, S2>> {
        &mut self.0
    }
}

impl<S1, S2> Clone for TensorOp<S1, S2>
where
    S1: RawDataClone,
    S2: RawDataClone,
{
    fn clone(&self) -> Self {
        TensorOp(self.0.clone())
    }
}

// impl<S1, S2> Copy for TensorOp<S1, S2>
// where
//     S1: RawDataClone,
//     S2: RawDataClone,
// {
// }

impl<S1, S2> Default for TensorOp<S1, S2>
where
    S1: RawData,
    S2: RawData,
{
    fn default() -> Self {
        TensorOp(None)
    }
}

impl<H, K> Deref for TensorOp<H, K>
where
    H: RawData,
    K: RawData,
{
    type Target = Option<TensorExpr<H, K>>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

impl<H, K> DerefMut for TensorOp<H, K>
where
    H: RawData,
    K: RawData,
{
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

impl<S1, S2> From<TensorOp<S1, S2>> for Option<TensorExpr<S1, S2>>
where
    S1: RawData,
    S2: RawData,
{
    fn from(op: TensorOp<S1, S2>) -> Self {
        op.0
    }
}

impl<S1, S2> From<Option<TensorExpr<S1, S2>>> for TensorOp<S1, S2>
where
    S1: RawData,
    S2: RawData,
{
    fn from(expr: Option<TensorExpr<S1, S2>>) -> Self {
        TensorOp(expr)
    }
}

impl<S1, S2> From<TensorExpr<S1, S2>> for TensorOp<S1, S2>
where
    S1: RawData,
    S2: RawData,
{
    fn from(expr: TensorExpr<S1, S2>) -> Self {
        TensorOp(Some(expr))
    }
}
