/*
    Appellation: ndtensor-traits <library>
    Created At: 2025.11.25:19:28:55
    Contrib: @FL03
*/
//! Common traits used to define the behavior of ndtensor library components
//! 

pub use self::{ops::*, raw_tensor::*, scalar::*};

pub mod ops;
pub mod raw_tensor;
pub mod scalar;

#[doc(hidden)]
pub mod prelude {
    //! A collection of commonly used traits from the `traits` module.
    pub use crate::ops::*;
    pub use crate::raw_tensor::*;
    pub use crate::scalar::*;
}