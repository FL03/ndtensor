/*
    Appellation: ndtensor <library>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
//! # ndtensor
//!
//!
#![crate_name = "ndtensor"]

extern crate acme;
extern crate ndarray as nd;

pub use self::{context::Context, errors::*, specs::*, tensor::*, types::*, utils::*};

pub(crate) mod context;
pub(crate) mod errors;
#[macro_use]
pub(crate) mod macros;
pub(crate) mod specs;
pub(crate) mod tensor;
pub(crate) mod utils;
#[macro_use]
pub mod ops;

pub(crate) mod impls {
    #[cfg(feature = "approx")]
    pub mod approx;
    pub mod create;
    pub mod grad;
    pub mod ops;
    pub mod reshape;

    pub mod views {
        pub mod dimensional;
        pub mod numerical;
        pub mod owned;
        pub mod raw;
        pub mod view;
    }
}

pub(crate) mod types {
    pub use self::{gradient::TensorGrad, kinds::*};

    pub(crate) mod gradient;
    pub(crate) mod kinds;
}

use ndarray::{CowRepr, IxDyn, OwnedArcRepr, OwnedRepr, ViewRepr};

pub type ArcTensor<A, D = IxDyn> = TensorBase<OwnedArcRepr<A>, D>;

pub type CowTensor<'a, A, D = IxDyn> = TensorBase<CowRepr<'a, A>, D>;

pub type RawTensorView<A, D = IxDyn> = TensorBase<ndarray::RawViewRepr<*const A>, D>;

pub type RawTensorViewMut<A, D = IxDyn> = TensorBase<ndarray::RawViewRepr<*mut A>, D>;

pub type Tensor<S, D = IxDyn> = TensorBase<OwnedRepr<S>, D>;

pub type TensorView<'a, S, D = IxDyn> = TensorBase<ViewRepr<&'a S>, D>;

pub type TensorViewMut<'a, S, D = IxDyn> = TensorBase<ViewRepr<&'a mut S>, D>;

pub type TensorId = acme::id::AtomicId;

pub type NdContainer<S> = ndarray::ArrayBase<S, ndarray::IxDyn>;

pub mod prelude {
    pub use crate::errors::{TensorError, TensorResult};
    pub use crate::ops::{TensorExpr, TensorOp};
    pub use crate::specs::*;
    pub use crate::tensor::TensorBase;
    pub use crate::types::*;
    pub use crate::utils::*;
    pub use crate::{
        ArcTensor, CowTensor, NdContainer, Tensor, TensorId, TensorView, TensorViewMut,
    };

    #[allow(unused_imports)]
    pub(crate) use ndarray::{
        array, s, ArrayBase, ArrayD, Data, DataOwned, Dimension, IxDyn, RawData, ShapeError,
    };
}
