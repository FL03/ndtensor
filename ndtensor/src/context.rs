/*
    Appellation: context <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::Mode;

use nd::{ArrayBase, Dimension, IntoDimension, RawData};

#[derive(Clone, Copy, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
pub struct Context {
    pub(crate) kind: Mode,
    pub(crate) rank: usize,
}

impl Context {
    pub fn new(kind: bool, rank: usize) -> Self {
        Self {
            kind: Mode::new(kind),
            rank,
        }
    }

    pub fn variable(mut self) -> Self {
        self.kind = Mode::Variable;
        self
    }

    pub fn from_shape<D>(shape: impl IntoDimension<Dim = D>) -> Self
    where
        D: Dimension,
    {
        Self::new(false, shape.into_dimension().ndim())
    }

    pub fn from_arr<S, D>(arr: ArrayBase<S, D>) -> Self
    where
        D: Dimension,
        S: RawData,
    {
        Self::new(false, arr.ndim())
    }

    pub fn into_var(self) -> Self {
        Self {
            kind: Mode::Variable,
            ..self
        }
    }

    pub fn kind(&self) -> Mode {
        self.kind
    }

    pub fn is_variable(&self) -> bool {
        self.kind().is_variable()
    }

    pub fn rank(&self) -> usize {
        self.rank
    }

    pub fn set_kind(&mut self, kind: Mode) {
        self.kind = kind;
    }

    pub fn set_rank(&mut self, rank: usize) {
        self.rank = rank;
    }
}
