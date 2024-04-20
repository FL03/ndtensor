/*
    Appellation: expr <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

use crate::TensorBase;

use acme::ops::{BinaryOp, UnaryOp};
use nd::{Data, DataMut, DataOwned, Dimension, OwnedArcRepr, OwnedRepr, ViewRepr};
use nd::{RawData, RawDataClone, RawDataMut, RawViewRepr};
use num::traits::{NumCast, ToPrimitive};

pub type BoxTensor<S> = Box<TensorBase<S>>;

pub enum ReshapeExpr<S, D>
where
    D: Dimension,
    S: RawData,
{
    Transpose(Box<Expr<S, D>>),
}

macro_rules! map_views {
    ($call:ident<$view:ident> where $($arg:ident:$($bound:ident)*),*) => {
        pub fn $call(self) -> TensorExpr<$view<A>, $view<B>> where $($arg: $($bound)*),* {
            fwd_view_body!(self, $call)
        }
    };
    (&$call:ident<$view:ident> where $($arg:ident:$($bound:ident)*),*) => {
        pub fn $call(&self) -> TensorExpr<$view<A>, $view<B>> where $($arg: $($bound)*),* {
            fwd_view_body!(&self, $call)
        }
    };
    (&mut $call:ident<$view:ident> where $($arg:ident:$($bound:ident)*),*) => {
        pub fn $call(&self) -> TensorExpr<$view<A>, $view<B>> where $($arg: $($bound)*),* {
            fwd_view_body!(&mut self, $call)
        }
    };
    ($call:ident<$view:ident>) => {
        pub fn $call(self) -> TensorExpr<$view<A>, $view<B>> {
            fwd_view_body!(self, $call)
        }
    };
    (&$call:ident<$view:ident>) => {
        pub fn $call(&self) -> TensorExpr<$view<A>, $view<B>> {
            fwd_view_body!(&self, $call)
        }
    };
    (&mut $call:ident<$view:ident>) => {
        pub fn $call(&self) -> TensorExpr<$view<A>, $view<B>> {
            fwd_view_body!(&mut self, $call)
        }
    };
}

pub enum Expr<S, D> where D: nd::Dimension, S: RawData, {
    Binary {
        lhs: Box<Expr<S, D>>,
        rhs: Box<Expr<S, D>>,
        op: BinaryOp,
    },
    Unary {
        recv: Box<Expr<S, D>>,
        op: UnaryOp,
    },
    Scalar(S::Elem),
    Tensor(TensorBase<S, D>),
}


pub enum TensorExpr<S1, S2 = S1>
where
    S1: RawData,
    S2: RawData,
{
    Binary {
        lhs: BoxTensor<S1>,
        rhs: BoxTensor<S2>,
        op: BinaryOp,
    },
    Unary {
        recv: BoxTensor<S1>,
        op: UnaryOp,
    },
    Matmul {
        lhs: BoxTensor<S1>,
        rhs: BoxTensor<S2>,
    },
    Transpose(BoxTensor<S1>),
}

impl<A, B, S1, S2> TensorExpr<S1, S2>
where
    S1: RawData<Elem = A>,
    S2: RawData<Elem = B>,
{
    pub fn binary(lhs: BoxTensor<S1>, rhs: BoxTensor<S2>, op: BinaryOp) -> Self {
        TensorExpr::Binary { lhs, rhs, op }
    }

    pub fn transpose(recv: BoxTensor<S1>) -> Self {
        TensorExpr::Transpose(recv)
    }

    pub fn unary(recv: BoxTensor<S1>, op: UnaryOp) -> Self {
        TensorExpr::Unary { recv, op }
    }

    pub fn numcast<C>(&self) -> TensorExpr<OwnedRepr<C>, OwnedRepr<C>>
    where
        A: Clone + ToPrimitive,
        B: Clone + ToPrimitive,
        C: NumCast,
        S1: Data,
        S2: Data,
    {
        fwd_expr_call!(&self.numcast::<C>().boxed())
    }

    map_views!(into_owned<OwnedRepr> where A: Clone, B: Clone, S1: DataOwned, S2: DataOwned);
    map_views!(into_shared<OwnedArcRepr> where S1: DataOwned, S2: DataOwned);

    map_views!(&to_owned<OwnedRepr> where A: Clone, B: Clone, S1: Data, S2: Data);
    map_views!(&to_shared<OwnedArcRepr> where A: Clone, B: Clone, S1: Data, S2: Data);

    pub fn raw_view(&self) -> TensorExpr<RawViewRepr<*const A>, RawViewRepr<*const B>> {
        fwd_view_body!(&self, raw_view)
    }

    pub fn raw_view_mut(&mut self) -> TensorExpr<RawViewRepr<*mut A>, RawViewRepr<*mut B>>
    where
        S1: RawDataMut,
        S2: RawDataMut,
    {
        fwd_view_body!(&mut self, raw_view_mut)
    }

    pub fn view(&self) -> TensorExpr<ViewRepr<&'_ A>, ViewRepr<&'_ B>>
    where
        S1: Data,
        S2: Data,
    {
        fwd_view_body!(&self, view)
    }

    pub fn view_mut(&mut self) -> TensorExpr<ViewRepr<&'_ mut A>, ViewRepr<&'_ mut B>>
    where
        S1: DataMut,
        S2: DataMut,
    {
        fwd_view_body!(&mut self, view_mut)
    }
}

impl<A, B, S1, S2> Clone for TensorExpr<S1, S2>
where
    S1: RawDataClone<Elem = A>,
    S2: RawDataClone<Elem = B>,
{
    fn clone(&self) -> Self {
        fwd_expr_call!(&self.clone())
    }
}
