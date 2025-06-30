/*
    Appellation: raw <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::{RawTensorView, RawTensorViewMut, TensorBase, TensorView};

macro_rules! impl_raw_tensor_view {
    ($($target:ident),*) => {
        $(
            impl_raw_tensor_view!(@impl $target);
        )*
    };
    (@impl $target:ident) => {
        impl<A, D, K> $target<A, D, K>
        where
            D: ndarray::Dimension,
            K: $crate::TensorMode,
        {
            pub unsafe fn cast<B>(self) -> $target<B, D, K> {
                TensorBase {
                    id: self.id,
                    ctx: self.ctx,
                    data: self.data.cast::<B>(),
                }
            }

            pub unsafe fn deref_into_view<'a>(self) -> TensorView<'a, A, D, K> {
                TensorBase {
                    id: self.id,
                    ctx: self.ctx,
                    data: self.data.deref_into_view(),
                }
            }
        }

    };
}

impl_raw_tensor_view!(RawTensorView, RawTensorViewMut);
