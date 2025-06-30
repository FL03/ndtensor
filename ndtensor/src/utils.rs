/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::error::InverseError;
use nd::ScalarOperand;
use nd::prelude::*;
use num::traits::{Float, NumAssign};

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



pub fn inverse<T>(matrix: &Array2<T>) -> Result<Array2<T>, InverseError>
where
    T: Copy + NumAssign + ScalarOperand,
{
    let (rows, cols) = matrix.dim();

    if !matrix.is_square() {
        return Err(InverseError::NonSquareMatrix); // Matrix must be square for inversion
    }

    let identity = Array2::eye(rows);

    // Construct an augmented matrix by concatenating the original matrix with an identity matrix
    let mut aug = Array2::zeros((rows, 2 * cols));
    aug.slice_mut(s![.., ..cols]).assign(matrix);
    aug.slice_mut(s![.., cols..]).assign(&identity);

    // Perform Gaussian elimination to reduce the left half to the identity matrix
    for i in 0..rows {
        let pivot = aug[[i, i]];

        if pivot == T::zero() {
            return Err(InverseError::SingularMatrix); // Matrix is singular
        }

        aug.slice_mut(s![i, ..]).mapv_inplace(|x| x / pivot);

        for j in 0..rows {
            if i != j {
                let am = aug.clone();
                let factor = aug[[j, i]];
                let rhs = am.slice(s![i, ..]);
                aug.slice_mut(s![j, ..])
                    .zip_mut_with(&rhs, |x, &y| *x -= y * factor);
            }
        }
    }

    // Extract the inverted matrix from the augmented matrix
    let inverted = aug.slice(s![.., cols..]);

    Ok(inverted.to_owned())
}