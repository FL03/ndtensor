/*
    Appellation: ndtensor <library>
    Contrib: @FL03
*/
//! # ndtensor
//!
//! Welcome to the `ndtensor` crate, a powerful N-dimensional tensor library for Rust.
//!
//! ### Features
//!
//! The crate supports various extensions or integrations that enhance its functionality. These
//! features can be enabled in your `Cargo.toml` file by specifying them under the `features`
//! section. Here are some of the available features:
//!
//! - `anyhow`: Enables the use of the `anyhow` crate for error handling.
//! - `approx`: Enables approximate equality checks for floating-point numbers.
//! - `complex`: Enables complex number support.
//! - `json`: Enables JSON serialization and deserialization capabilities.
//! - `rand`: Enables random number generation capabilities.
//! - `serde`: Enables serialization and deserialization capabilities.
//! - `tracing`: Enables tracing capabilities for debugging and logging.
//!
#![allow(
    clippy::missing_safety_doc,
    clippy::module_inception,
    clippy::needless_doctest_main,
    clippy::upper_case_acronyms
)]
#![cfg_attr(not(feature = "std"), no_std)]
#![crate_type = "lib"]

#[cfg(not(all(feature = "std", feature = "alloc")))]
compiler_error! {
    "Either the `std` or `alloc` feature must be enabled for this crate to complie."
}

#[macro_use]
pub(crate) mod macros {
    #[macro_use]
    pub mod seal;
}

#[cfg(feature = "alloc")]
extern crate alloc;

#[doc(inline)]
pub use ndtensor_traits as traits;
#[doc(inline)]
pub use ndtensor_traits::prelude::*;

#[doc(inline)]
pub use self::{error::*, tensor::*, types::*};

/// this module defines the [`TensorError`] enum for handling tensor-related errors
pub mod error;
/// this module defines various iterators for the [`TensorBase`]
pub mod iter;

mod tensor;

mod impls {
    mod impl_tensor;
    mod impl_tensor_iter;
    mod impl_tensor_ops;
    mod impl_tensor_repr;

    #[allow(deprecated)]
    mod impl_tensor_deprecated;
    #[cfg(feature = "rand")]
    mod impl_tensor_rand;
    #[cfg(feature = "serde")]
    mod impl_tensor_serde;
}

mod types {
    //! this module defines various type aliases and primitives used by the `tensor` module
    #[doc(inline)]
    pub use self::prelude::*;

    mod aliases;

    mod prelude {
        #[doc(inline)]
        pub use super::aliases::*;
    }
}

#[doc(hidden)]
pub mod prelude {
    #[doc(inline)]
    pub use ndtensor_traits::prelude::*;

    #[doc(inline)]
    pub use super::tensor::*;
    #[doc(inline)]
    pub use super::types::*;
}
