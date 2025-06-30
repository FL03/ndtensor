/*
    Appellation: misc <impls>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::CowTensor;

impl<'a, A, D, K> CowTensor<'a, A, D, K>
where
    D: Dimension,
{
    pub fn is_view(&self) -> bool {
        self.borrow().is_view()
    }
}