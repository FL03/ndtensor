/*
    Appellation: ndtensor <test>
    Created At: 2025.11.25:19:38:57
    Contrib: @FL03
*/
use ndtensor_traits::NdTensorExt;

use ndarray::Array2;

#[test]
fn test_ndtensor_constructors() {
    const fn generator((x, y): (usize, usize)) -> f64 {
        x as f64 / (x + y) as f64
    }
    let a: Array2<f64> = Array2::<f64>::from_shape_with_fn([2, 3], generator);
    assert_eq!(a.shape(), &[2, 3]);
}
