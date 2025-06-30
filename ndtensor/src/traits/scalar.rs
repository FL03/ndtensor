/*
    Appellation: scalar <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::Tensor;
use core::iter::{Product, Sum};
use core::ops::Neg;
use nd::Ix0;
use num::traits::{Num, NumOps, Pow};

pub trait Scalar:
    Copy + Neg<Output = Self> + Num + Pow<Self, Output = Self> + Product + Sized + Sum + 'static
{
    type Imag: Scalar + NumOps<Self::Real, Self::Imag>;
    type Real: Scalar + NumOps<Self::Imag, Self::Imag>;

    fn abs(self) -> Self;

    fn acos(self) -> Self;

    fn acosh(self) -> Self;

    fn asin(self) -> Self;

    fn asinh(self) -> Self;

    fn atan(self) -> Self;

    fn atanh(self) -> Self;

    fn cube(self) -> Self;

    fn cbrt(self) -> Self;

    fn conj(self) -> Self;

    fn cos(self) -> Self;

    fn cosh(self) -> Self;

    fn exp(self) -> Self;

    fn ln(self) -> Self;

    fn log(self, base: Self) -> Self;

    fn powi(self, n: i32) -> Self;

    fn recip(self) -> Self;

    fn sin(self) -> Self;

    fn sinh(self) -> Self;

    fn sqrd(self) -> Self;

    fn sqrt(self) -> Self;

    fn tan(self) -> Self;

    fn tanh(self) -> Self;
}
pub trait ScalarExt: Scalar {
    fn into_tensor(self) -> Tensor<Self, Ix0> {
        Tensor::from_scalar(self)
    }

    fn sigmoid(self) -> Self {
        (Self::one() + self.neg().exp()).recip()
    }
}

impl<S> ScalarExt for S where S: Scalar {}
