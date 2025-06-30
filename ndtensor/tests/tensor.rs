/*
    Appellation: Tensor <test>
    Contrib: @FL03
*/
use ndtensor::Tensor;

#[test]
fn test_ones_and_zeros() {
    // weights retain the given shape (d_in, d_out)
    // bias retains the shape (d_out,)
    let ones = Tensor::<f64>::ones((3, 4));
    assert_eq!(ones.dim(), (3, 4));
    // weights retain the given shape (d_in, d_out)
    // bias retains the shape (d_out,)
    let zeros = Tensor::<f64>::zeros((3, 4));
    assert_eq!(zeros.dim(), (3, 4));
}
