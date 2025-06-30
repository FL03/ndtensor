/*
    appellation: impl_tensor <module>
    authors: @FL03
*/
use crate::tensor::TensorBase;

use crate::Tensor;
use ndarray::{ArrayBase, Data, Dimension, RawData, RawDataClone};
use num_complex::ComplexFloat;
use num_traits::Float;

impl<A, S, D> TensorBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
}

#[allow(unused_macros)]
macro_rules! unary {
    (@impl $(#[doc = $doc:literal])? $method:ident($f:expr) -> $out:ty $(where $($rest:tt)*)?) => {
        $(#[doc = $doc])?
        pub fn $method(&self) -> $out $(where $($rest)*)? {
            self.map(|x| $f(x))
        }
    };
    ($(
        $(#[doc = $doc:literal])?
        $method:ident($f:expr) -> $out:ty $(where $($rest:tt)*)?
    );* $(;)?) => {
        $(
            unary!{ @impl
                $(#[doc = $doc])?
                $method($f) -> $out $(where $($rest)*)?
            }
        )*
    };
}

impl<A, S, D> TensorBase<S, D>
where
    A: Float,
    D: Dimension,
    S: Data<Elem = A>,
{
    /// returns the conjugate of the tensor; the conjugate of all real numbers is itself, while
    /// the conjugate of an imaginary number is found by negating the imaginary part:
    ///
    /// ```math
    /// \mbox{conj}(a + bi) = a - bi
    /// ```
    pub fn conj(&self) -> Tensor<A, D>
    where
        A: ComplexFloat,
    {
        self.map(|x| x.conj())
    }
    /// this method applies the L1 normalization technique to every element within the tensor.
    /// This is primarily used for normalizing complex numbers, where the L1 norm is defined as
    /// the sum of the absolute values of the real and imaginary parts.
    pub fn norm_c(&self) -> Tensor<A::Real, D>
    where
        A: ComplexFloat,
    {
        self.map(|x| x.l1_norm())
    }

    unary! {
        #[doc = "compute the absolute value of every element within the tensor"]
        abs(A::abs) -> Tensor<A, D>;
        #[doc = "compute the cosine of every element within the tensor"]
        cos(A::cos) -> Tensor<A, D>;
        #[doc = "compute the hyperbolic cosine of every element within the tensor"]
        cosh(A::cosh) -> Tensor<A, D>;
        #[doc = "compute the sine of every element within the tensor"]
        sin(A::sin) -> Tensor<A, D>;
        #[doc = "compute the hyperbolic sine of every element within the tensor"]
        sinh(A::sinh) -> Tensor<A, D>;
        #[doc = "compute the tangent of every element within the tensor"]
        tan(A::tan) -> Tensor<A, D>;
        #[doc = "compute the hyperbolic tangent of every element within the tensor"]
        tanh(A::tanh) -> Tensor<A, D>;
        #[doc = "compute the exponential of every element within the tensor"]
        exp(A::exp) -> Tensor<A, D>;
        #[doc = "compute the logarithm base 10 of every element within the tensor"]
        log10(A::log10) -> Tensor<A, D>;
        #[doc = "compute the logarithm base 2 of every element within the tensor"]
        log2(A::log2) -> Tensor<A, D>;
        #[doc = "compute the natural logarithm of every element within the tensor"]
        ln(A::ln) -> Tensor<A, D>;
        #[doc = "compute the square root of every element within the tensor"]
        sqrt(A::sqrt) -> Tensor<A, D>;
    }
}

impl<A, S, D> Clone for TensorBase<S, D>
where
    A: Clone,
    S: RawDataClone<Elem = A>,
    D: Dimension,
{
    fn clone(&self) -> Self {
        TensorBase {
            store: self.store().clone(),
        }
    }
}

impl<A, S, D> Copy for TensorBase<S, D>
where
    A: Copy,
    S: RawDataClone<Elem = A> + Copy,
    D: Dimension + Copy,
{
}

impl<A, S, D> core::fmt::Debug for TensorBase<S, D>
where
    A: core::fmt::Debug,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.debug_struct("TensorBase")
            .field("store", &self.store())
            .finish()
    }
}

impl<A, S, D> core::fmt::Display for TensorBase<S, D>
where
    A: core::fmt::Display,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        write!(f, "{}", self.store())
    }
}

impl<A, S, D> PartialEq for TensorBase<S, D>
where
    A: PartialEq,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn eq(&self, other: &Self) -> bool {
        self.store() == other.store()
    }
}

impl<A, S, D> PartialEq<ArrayBase<S, D>> for TensorBase<S, D>
where
    A: PartialEq,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn eq(&self, other: &ArrayBase<S, D>) -> bool {
        self.store() == other
    }
}

impl<A, S, D> PartialEq<&ArrayBase<S, D>> for TensorBase<S, D>
where
    A: PartialEq,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn eq(&self, other: &&ArrayBase<S, D>) -> bool {
        self.store() == *other
    }
}

impl<A, S, D> PartialEq<&mut ArrayBase<S, D>> for TensorBase<S, D>
where
    A: PartialEq,
    S: Data<Elem = A>,
    D: Dimension,
{
    fn eq(&self, other: &&mut ArrayBase<S, D>) -> bool {
        self.store() == *other
    }
}
