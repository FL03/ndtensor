/*
    Appellation: misc <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::CowTensor;

impl<'a, A, D> crate::CowTensor<'a, A, D>
where
    D: Dimension,
{
    pub fn is_view(&self) -> bool {
        self.data().is_view()
    }
}