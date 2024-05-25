/*
    Appellation: ops <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{expr::*, wrapper::TensorOp};

pub(crate) mod expr;
pub(crate) mod wrapper;

use nd::*;

pub type OwnedOp<A, B> = TensorOp<OwnedRepr<A>, OwnedRepr<B>>;
pub type ArcOp<A, B> = TensorOp<OwnedArcRepr<A>, OwnedArcRepr<B>>;

pub trait NdTensorOp<S1, S2>
where
    S1: RawData,
    S2: RawData,
{
    fn from_expr(expr: TensorExpr<S1, S2>) -> Self;
    fn new(expr: Option<TensorExpr<S1, S2>>) -> Self;
    fn none() -> Self;
    ///
    fn is_none(&self) -> bool;
    ///
    fn is_some(&self) -> bool;

    fn as_ref(&self) -> Option<&TensorExpr<S1, S2>>;

    fn as_mut(&mut self) -> Option<&mut TensorExpr<S1, S2>>;
}

pub trait NdTensorOpExt<S1, S2>: NdTensorOp<S1, S2>
where
    S1: RawData,
    S2: RawData,
{
    type Op: NdTensorOp<S1, S2>;

    fn map<F, O, T1>(self, f: F) -> O
    where
        O: NdTensorOp<T1, T1> + Sized,
        F: FnOnce(TensorExpr<S1, S2>) -> TensorExpr<T1>,
        T1: RawData;
}

impl<A, B, S1, S2> NdTensorOp<S1, S2> for Option<TensorExpr<S1, S2>>
where
    S1: RawData<Elem = A>,
    S2: RawData<Elem = B>,
{
    fn from_expr(expr: TensorExpr<S1, S2>) -> Self {
        Some(expr)
    }

    fn new(expr: Option<TensorExpr<S1, S2>>) -> Self {
        expr
    }

    fn none() -> Self {
        None
    }

    fn is_none(&self) -> bool {
        self.is_none()
    }

    fn is_some(&self) -> bool {
        self.is_some()
    }

    fn as_ref(&self) -> Option<&TensorExpr<S1, S2>> {
        self.as_ref()
    }

    fn as_mut(&mut self) -> Option<&mut TensorExpr<S1, S2>> {
        self.as_mut()
    }
}

macro_rules! apply_raw {
    ($(*$id:ident),*) => {
        $(
            apply_raw!(@impl *$id);
        )*
    };
    (@impl *$id:ident) => {
        impl<A, B> TensorExpr<RawViewRepr<*$id A>, RawViewRepr<*$id B>> {
            pub unsafe fn cast<C>(self) -> TensorExpr<RawViewRepr<*$id C>, RawViewRepr<*$id C>> {
                fwd_expr_call!(self.cast().boxed())
            }

            pub unsafe fn deref_into_view<'a>(self) -> TensorExpr<ViewRepr<&'a A>, ViewRepr<&'a B>> {
                fwd_expr_call!(self.deref_into_view().boxed())
            }
        }

        impl<A, B> TensorOp<RawViewRepr<*$id A>, RawViewRepr<*$id B>> {
            pub unsafe fn cast<C>(self) -> TensorOp<RawViewRepr<*$id C>, RawViewRepr<*$id C>> {
                TensorOp(self.0.map(|expr| expr.cast()))
            }

            pub unsafe fn deref_into_view<'a>(self) -> TensorOp<ViewRepr<&'a A>, ViewRepr<&'a B>> {
                TensorOp(self.0.map(|expr| expr.deref_into_view()))
            }
        }
    };

}


macro_rules! fwd_expr_call {
    ($self:ident.$($rest:tt)*) => {
        match $self {
            $crate::grad::ops::TensorExpr::Binary { lhs, rhs, op } => $crate::grad::ops::TensorExpr::Binary {
                lhs: lhs.$($rest)*,
                rhs: rhs.$($rest)*,
                op,
            },
            $crate::grad::ops::TensorExpr::Unary { recv, op } => $crate::grad::ops::TensorExpr::Unary {
                recv: recv.$($rest)*,
                op,
            },
            $crate::grad::ops::TensorExpr::Matmul { lhs, rhs } => $crate::grad::ops::TensorExpr::Matmul {
                lhs: lhs.$($rest)*,
                rhs: rhs.$($rest)*,
            },
            $crate::grad::ops::TensorExpr::Transpose(recv) => $crate::grad::ops::TensorExpr::Transpose(recv.$($rest)*),
        }
    };
    (&$self:ident.$($rest:tt)*) => {
        match $self {
            $crate::grad::ops::TensorExpr::Binary { lhs, rhs, op } => $crate::grad::ops::TensorExpr::Binary {
                lhs: lhs.$($rest)*,
                rhs: rhs.$($rest)*,
                op: *op,
            },
            $crate::grad::ops::TensorExpr::Unary { recv, op } => $crate::grad::ops::TensorExpr::Unary {
                recv: recv.$($rest)*,
                op: *op,
            },
            $crate::grad::ops::TensorExpr::Matmul { lhs, rhs } => $crate::grad::ops::TensorExpr::Matmul {
                lhs: lhs.$($rest)*,
                rhs: rhs.$($rest)*,
            },
            $crate::grad::ops::TensorExpr::Transpose(recv) => $crate::grad::ops::TensorExpr::Transpose(recv.$($rest)*),
        }
    };
}


apply_raw!(*const, *mut);
