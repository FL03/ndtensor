/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use nd::prelude::*;
use num::Float;

/// Hashes a dimension using the [DefaultHasher].
#[cfg(feature = "std")]
pub fn hash_dim<D, Sh>(shape: Sh) -> u64
where
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    use std::hash::{DefaultHasher, Hash, Hasher};
    let shape = shape.into_shape();
    let dim = shape.raw_dim().clone();
    let mut s = DefaultHasher::new();
    for i in dim.slice() {
        i.hash(&mut s);
    }
    s.finish()
}

/// Generates a new [Array] using evenly spaced values between [0, n-1);
/// where n is the product of the dimensions.
pub fn linarr<A, D, Sh>(shape: Sh) -> Array<A, D>
where
    A: Float,
    D: Dimension,
    Sh: ShapeBuilder<Dim = D>,
{
    let shape = shape.into_shape();
    let dim = shape.raw_dim().clone();
    let dview = dim.as_array_view();
    let n = dview.product();
    Array::linspace(A::zero(), A::from(n).unwrap() - A::one(), n)
        .into_shape(dim)
        .expect("linspace err")
}
