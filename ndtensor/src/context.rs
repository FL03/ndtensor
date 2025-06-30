/*
    Appellation: context <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::types::{Mode, Normal, TensorMode, Variable};

use core::marker::PhantomData;
use nd::{ArrayBase, Dimension, IntoDimension, RawData};

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Context<K = Variable> {
    pub(crate) rank: usize,
    _kind: PhantomData<K>,
}

impl<K> Context<K>
where
    K: TensorMode,
{
    pub fn new(rank: usize) -> Self {
        Self {
            rank,
            _kind: PhantomData::<K>,
        }
    }

    pub fn detach(&self) -> Context<Normal> {
        Context {
            rank: self.rank,
            _kind: PhantomData::<Normal>,
        }
    }

    pub fn into_normal(self) -> Context<Normal> {
        Context {
            rank: self.rank,
            _kind: PhantomData::<Normal>,
        }
    }

    pub fn into_variable(self) -> Context<Variable> {
        Context {
            rank: self.rank,
            _kind: PhantomData::<Variable>,
        }
    }

    pub fn from_shape<D>(shape: impl IntoDimension<Dim = D>) -> Self
    where
        D: Dimension,
    {
        Self::new(shape.into_dimension().ndim())
    }

    pub fn from_arr<S, D>(arr: &ArrayBase<S, D>) -> Self
    where
        D: Dimension,
        S: RawData,
    {
        Self::new(arr.ndim())
    }

    pub fn kind(&self) -> Mode {
        Mode::new::<K>()
    }

    pub fn is_variable(&self) -> bool {
        Variable::is::<K>()
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn into_kind<J>(self) -> Context<J>
    where
        J: TensorMode,
    {
        Context::<J>::new(self.rank())
    }

    pub fn set_rank(&mut self, rank: usize) {
        self.rank = rank;
    }
}
