/*
    Appellation: traits <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::prelude::*;

pub mod build;
pub mod convert;
pub mod ndtensor;
pub mod scalar;
pub mod shape;

pub(crate) mod prelude {
    pub use super::build::*;
    pub use super::convert::*;
    pub use super::ndtensor::*;
    pub use super::scalar::*;
    pub use super::shape::*;
}
