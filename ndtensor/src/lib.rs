/*
    Appellation: ndtensor <library>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # ndtensor
//!
//!
#![crate_name = "ndtensor"]
#![cfg_attr(not(feature = "std"), no_std)]

extern crate acme;
#[cfg(feature = "alloc")]
extern crate alloc;
extern crate ndarray as nd;

pub use self::{context::Context, tensor::TensorBase, traits::prelude::*, types::*, utils::*};
pub use self::error::{TensorError, TensorResult};

#[macro_use]
pub(crate) mod macros;
pub(crate) mod context;
pub(crate) mod tensor;
pub(crate) mod utils;



pub mod error;
#[doc(hidden)]
pub mod grad;
pub mod traits;

mod impls {
    #[cfg(feature = "approx")]
    mod approx;
    mod create;
    mod linalg;
    mod ops;
    mod reshape;
    mod tensor;

    mod views {
        mod dimensional;
        mod numerical;
        mod owned;
        mod raw;
        mod view;
    }
}

pub(crate) mod types {
    pub use self::{gradient::TensorGrad, kinds::*};

    pub(crate) mod gradient;
    pub(crate) mod kinds;

    pub(crate) mod prelude {
        pub use super::gradient::TensorGrad;
        pub use super::kinds::*;
    }
}

macro_rules! tensor_ref {
    ($($name:ident<$S:ident>$(($($rest:tt)*))? ),*) => {
        $(
            tensor_ref!(@impl $name<$S> $(:$($rest)*)?);
        )*
    };

    (@impl $name:ident<$S:ident>) => {
        pub type $name<A = f64, D = ndarray::IxDyn, K = $crate::types::Normal> = $crate::tensor::TensorBase<ndarray::$S<A>, D, K>;
    };
    (@impl $name:ident<$S:ident>: 'a) => {
        pub type $name<'a, A = f64, D = ndarray::IxDyn, K = $crate::types::Normal> = $crate::tensor::TensorBase<ndarray::$S<'a, A>, D, K>;
    };
    (@impl $name:ident<$S:ident>: &'a) => {
        pub type $name<'a, A = f64, D = ndarray::IxDyn, K = $crate::types::Normal> = $crate::tensor::TensorBase<ndarray::$S<&'a A>, D, K>;
    };
    (@impl $name:ident<$S:ident>: &'a mut) => {
        pub type $name<'a, A = f64, D = ndarray::IxDyn, K = $crate::types::Normal> = $crate::tensor::TensorBase<ndarray::$S<&'a mut A>, D, K>;
    };
    (@impl $name:ident<$S:ident>: *$ptr:ident) => {
        pub type $name<A = f64, D = ndarray::IxDyn, K = $crate::types::Normal> = $crate::tensor::TensorBase<ndarray::$S<*$ptr A>, D, K>;
    };
}

tensor_ref! {
    ArcTensor<OwnedArcRepr>,
    CowTensor<CowRepr>('a),
    RawTensorView<RawViewRepr>(*const),
    RawTensorViewMut<RawViewRepr>(*mut),
    Tensor<OwnedRepr>,
    TensorView<ViewRepr>(&'a),
    TensorViewMut<ViewRepr>(&'a mut)
}

pub type TensorId = acme::id::AtomicId;

pub mod prelude {
    pub use crate::error::prelude::*;
    pub use crate::tensor::TensorBase;
    pub use crate::traits::prelude::*;
    pub use crate::types::prelude::*;
    pub use crate::utils::*;
    pub use crate::{ArcTensor, CowTensor, Tensor, TensorId, TensorView, TensorViewMut};

    #[allow(unused_imports)]
    pub(crate) use ndarray::{
        array, s, ArrayBase, ArrayD, Data, DataOwned, Dimension, IxDyn, RawData, ShapeError,
    };
}
