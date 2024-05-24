/*
    Appellation: tensor <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
extern crate ndtensor;

use approx::AbsDiffEq;
use ndarray::prelude::*;
use ndtensor::Tensor;

#[test]
fn test_abs_diff_eq() {
    let tensor = Tensor::<f64, Ix2>::linshape((2, 2)).unwrap();
    let res = tensor.cos();
    assert!(res.abs_diff_eq(&tensor.data().mapv(|i| i.cos()), 1e-8));
}
