/*
    Appellation: scalar <module>
    Contrib: @FL03
*/
use num_traits::Float;

#[cfg(feature = "complex")]
use num_complex::ComplexFloat;

/// The [`Number`] trait is a sealed trait used to generically define behaviors common to all
/// numerical primitives (i.e., u8, i32, f64, etc.) as well as composite numerical types
/// (e.g., arrays, tensors, etc.) that encapsulate numerical primitives.
pub trait Number
where
    Self: 'static
        + Copy
        + Default
        + PartialEq
        + PartialOrd
        + core::fmt::Debug
        + core::iter::Product
        + core::iter::Sum
        + core::ops::Add<Output = Self>
        + core::ops::Div<Output = Self>
        + core::ops::Mul<Output = Self>
        + core::ops::Sub<Output = Self>
        + core::ops::Rem<Output = Self>
        + core::ops::AddAssign
        + core::ops::DivAssign
        + core::ops::MulAssign
        + core::ops::RemAssign
        + core::ops::SubAssign,
{
    private! {}
}

pub trait Scalar {
    private! {}
}

#[cfg(feature = "complex")]
pub trait ScalarComplex<T> {
    type Complex<U>: ComplexFloat<Real = U>;

    private!();
    /// create a new complex number
    fn new(real: T, imag: T) -> Self::Complex<T>;
    /// returns a reference to the real part of the object
    fn real(&self) -> T;
    /// returns a reference to the imaginary part of the object
    fn imag(&self) -> T;
    /// returns the absolute value of the complex number
    fn abs(&self) -> T
    where
        T: Float,
    {
        self.real().hypot(self.imag())
    }
    /// compute the complex conjugate of the object
    fn conj(&self) -> Self::Complex<T>
    where
        T: core::ops::Neg<Output = T>,
    {
        Self::new(self.real(), self.imag().neg())
    }
}

/*
 ************* Implementations *************
*/
macro_rules! impl_number {
    ($($T:ty),* $(,)?) => {
        $(
            impl Number for $T {
                seal! {}
            }
        )*
    };
}

macro_rules! impl_scalar {
    ($($T:ty),* $(,)?) => {
        $(
            impl Scalar for $T {
                seal! {}
            }
        )*
    };
}

impl_number! {
    u8, u16, u32, u64, u128, usize,
    i8, i16, i32, i64, i128, isize,
    f32, f64,
}

impl_scalar! {
    u8, u16, u32, u64, u128, usize,
    i8, i16, i32, i64, i128, isize,
    f32, f64,
}
