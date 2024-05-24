/*
    Appellation: tensor <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(test)]

extern crate ndarray as nd;
extern crate ndtensor;

use ndarray::prelude::*;
use ndtensor::{Tensor, Variable};

#[test]
fn test_tensor() {
    let tensor: Tensor<f64, IxDyn, Variable> =
        Tensor::ndtensor(nd::array![[0f64, 1f64], [2f64, 3f64]]);

    assert!(tensor.op().is_none());
}

#[test]
fn test_index() {
    let tensor = Tensor::<f64, nd::Ix3>::linshape((3, 3, 3)).unwrap();

    assert_eq!(tensor[[0, 0, 0]], 0f64);
}
