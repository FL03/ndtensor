/*
    Appellation: error <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
pub use self::{err::*, kinds::*};

pub(crate) mod err;

pub type TensorResult<T = ()> = core::result::Result<T, TensorError>;

pub mod kinds {
    pub use self::prelude::*;

    pub(crate) mod inverse;

    pub(crate) mod prelude {
        pub use super::inverse::*;
    }
}

pub(crate) mod prelude {    
    pub use super::TensorResult;
    pub use super::err::*;
    pub use super::kinds::prelude::*;
}  