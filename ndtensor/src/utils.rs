/*
    Appellation: utils <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::{TensorExpr, TensorId};
use crate::TensorBase;
use nd::{Array, Dimension, IntoDimension, RawData, RawDataClone};
use num::Float;
use std::collections::HashMap;

/// Hashes a dimension using the [DefaultHasher].
#[cfg(feature = "std")]
pub fn hash_dim<D>(dim: impl IntoDimension<Dim = D>) -> u64
where
    D: Dimension,
{
    use std::hash::{DefaultHasher, Hash, Hasher};
    let dim = dim.into_dimension();
    let mut s = DefaultHasher::new();
    for i in dim.slice() {
        i.hash(&mut s);
    }
    s.finish()
}

pub fn linarr<A, D>(dim: impl IntoDimension<Dim = D>) -> Array<A, D>
where
    A: Float,
    D: Dimension,
{
    let dim = dim.into_dimension();
    let dview = dim.as_array_view();
    let n = dview.product();
    Array::linspace(A::zero(), A::from(n).unwrap() - A::one(), n)
        .into_shape(dim)
        .expect("linspace err")
}

pub(crate) fn walk<S>(
    scope: TensorBase<S>,
    nodes: Vec<TensorBase<S>>,
    visited: &mut HashMap<TensorId, bool>,
) -> (bool, Vec<TensorBase<S>>)
where
    S: RawData + RawDataClone,
{
    if let Some(&tg) = visited.get(&scope.id()) {
        return (tg, nodes);
    }
    // track the gradient of the current node
    let mut track = false;
    // recursively call on the children nodes
    let mut nodes = if scope.is_variable() {
        // Do not call recursively on the "leaf" nodes.
        track = true;
        nodes
    } else if let Some(op) = scope.op() {
        match op {
            TensorExpr::Binary { lhs, rhs, .. } => {
                let (tg, nodes) = walk(*lhs.clone(), nodes, visited);
                track |= tg;
                let (tg, nodes) = walk(*rhs.clone(), nodes, visited);
                track |= tg;
                nodes
            }
            TensorExpr::Unary { recv, .. } => {
                let (tg, nodes) = walk(*recv.clone(), nodes, visited);
                track |= tg;
                nodes
            }
            _ => nodes,
        }
    } else {
        nodes
    };
    visited.insert(scope.id(), track);
    if track {
        nodes.push(scope);
    }
    (track, nodes)
}
