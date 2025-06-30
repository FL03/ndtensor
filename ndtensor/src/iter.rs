/*
    appellation: iter <module>
    authors: @FL03
*/
#[cfg(feature = "rayon")]
pub use self::parallel::*;

#[cfg(feature = "rayon")]
mod parallel;

use ndarray::Dimension;
use ndarray::iter::{Iter as NdIter, IterMut as NdIterMut};

/// An iterator over the elements of an n-dimensional tensor.
pub struct Iter<'a, A, D>
where
    D: Dimension,
{
    pub(crate) iter: NdIter<'a, A, D>,
}
/// A mutable iterator over the elements of an n-dimensional tensor.
pub struct IterMut<'a, A, D>
where
    D: Dimension,
{
    pub(crate) iter: NdIterMut<'a, A, D>,
}

/*
 ************* Implementations *************
*/

impl<'a, A, D> Iterator for Iter<'a, A, D>
where
    D: Dimension,
{
    type Item = &'a A;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}

impl<'a, A, D> Iterator for IterMut<'a, A, D>
where
    D: Dimension,
{
    type Item = &'a mut A;

    fn next(&mut self) -> Option<Self::Item> {
        self.iter.next()
    }
}
