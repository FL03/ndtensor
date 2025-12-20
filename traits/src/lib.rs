/*
    Appellation: ndtensor-traits <library>
    Created At: 2025.11.25:19:28:55
    Contrib: @FL03
*/
//! Common traits working to generalize the behaviors of an n-dimensional tensor.
#![allow(
    clippy::missing_safety_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::should_implement_trait,
    clippy::upper_case_acronyms,
    rustdoc::redundant_explicit_links
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![cfg_attr(feature = "nightly", feature(allocator_api))]
#![crate_type = "lib"]

#[cfg(not(any(feature = "std", feature = "alloc")))]
compiler_error! {
    "At least one of the \"std\" or \"alloc\" features must be enabled for the crate to compile."
}

#[cfg(feature = "alloc")]
extern crate alloc;

#[macro_use]
pub(crate) mod macros {
    #[macro_use]
    pub mod seal;
}
// modules
pub mod ndtensor;
pub mod ops;
pub mod raw_tensor;
pub mod scalar;
// re-exports
#[doc(inline)]
pub use self::prelude::*;
// prelude
#[doc(hidden)]
pub mod prelude {
    //! A collection of commonly used traits from the `traits` module.
    pub use crate::ndtensor::*;
    pub use crate::ops::*;
    pub use crate::scalar::*;
}
