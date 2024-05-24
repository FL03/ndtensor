/*
    Appellation: convert <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::TensorBase;
use nd::{Dimension, RawData};

pub trait IntoTensor<S, D>
where
    D: Dimension,
    S: RawData,
{
    fn into_tensor(self) -> TensorBase<S, D>;
}

/*
 ************* Implementations *************
*/
impl<S, D, U> IntoTensor<S, D> for U
where
    D: Dimension,
    S: RawData,
    U: Into<TensorBase<S, D>>,
{
    fn into_tensor(self) -> TensorBase<S, D> {
        self.into()
    }
}
