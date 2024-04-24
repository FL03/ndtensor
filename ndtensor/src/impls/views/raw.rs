/*
    Appellation: raw <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::ops::{TensorExpr, TensorOp};
use crate::{RawTensorView, RawTensorViewMut, TensorBase, TensorView};
use nd::{Dimension, RawViewRepr, ViewRepr};

macro_rules! impl_raw_tensor_view {
    ($($target:ident),*) => {
        $(
            impl_raw_tensor_view!(@impl $target);
        )*
    };
    (@impl $target:ident) => {
        impl<A, D> $target<A, D>
        where
            D: Dimension,
        {
            pub unsafe fn cast<B>(self) -> $target<B, D> {
                TensorBase {
                    id: self.id,
                    ctx: self.ctx,
                    data: self.data.cast::<B>(),
                    op: self.op.cast(),
                }
            }

            pub unsafe fn deref_into_view<'a>(self) -> TensorView<'a, A, D> {
                TensorBase {
                    id: self.id,
                    ctx: self.ctx,
                    data: self.data.deref_into_view(),
                    op: self.op.deref_into_view(),
                }
            }
        }

    };
}

impl_raw_tensor_view!(RawTensorView, RawTensorViewMut);

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

apply_raw!(*const, *mut);
