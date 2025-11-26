/*
    Appellation: ndtensor <test>
    Created At: 2025.11.25:19:38:57
    Contrib: @FL03
*/
use ndtensor_traits::NdTensor;

use ndarray::{Array2, Ix2};

#[test]
fn test_ndtensor_constructors() {
    let a: Array2<f64> = NdTensor::<f64, Ix2>::zeros(&[2, 3]);
    assert_eq!(a.shape(), &[2, 3]);
}