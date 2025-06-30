/*
    appellation: impl_tensor_rand <module>
    authors: @FL03
*/
use crate::tensor::TensorBase;

use ndarray::{DataOwned, Dimension, ShapeBuilder};
use rand::RngCore;
use rand_distr::Distribution;

impl<A, S, D> TensorBase<S, D>
where
    D: Dimension,
    S: DataOwned<Elem = A>,
{
    /// generate a new tensor with the given shape and randomly initialized values
    pub fn random<Sh, Ds>(shape: Sh, distr: Ds) -> Self
    where
        Ds: Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
    {
        use rand::{SeedableRng, rngs::SmallRng};
        Self::random_using(shape, distr, &mut SmallRng::from_rng(&mut rand::rng()))
    }
    /// randomly initializes a tensor with the given shape using the provided distribution and 
    /// random number generator
    pub fn random_using<Sh, Ds, R>(shape: Sh, distr: Ds, rng: &mut R) -> Self
    where
        R: RngCore + ?Sized,
        Ds: Distribution<A>,
        Sh: ShapeBuilder<Dim = D>,
    {
        Self::from_shape_fn(shape, |_| distr.sample(rng))
    }
    /// Randomly initializes a tensor with the given shape using the provided distribution
    pub fn random_with<Dst, Sh>(shape: Sh, distr: Dst) -> Self
    where
        S: DataOwned,
        Sh: ShapeBuilder<Dim = D>,
        Dst: Clone + Distribution<A>,
    {
        Self::random_using(shape, distr, &mut rand::rng())
    }
}
