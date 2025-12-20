/*
    Appellation: Tensor <test>
    Contrib: @FL03
*/
use ndtensor::Tensor;

#[test]
fn test_ones_and_zeros() {
    let dim = (3, 4);
    // ensure that the created tensors have the correct shape
    let ones = Tensor::<f64>::ones(dim);
    assert_eq! { ones.dim(), dim }
    assert! { ones.iter().all(|&x| x == 1f64) }
    // weights retain the given shape (d_in, d_out)
    // bias retains the shape (d_out,)
    let zeros = Tensor::<f64>::zeros(dim);
    assert_eq! { zeros.dim(), dim }
    assert! { zeros.iter().all(|&x| x == 0f64) }
}
