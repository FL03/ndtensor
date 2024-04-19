/*
    Appellation: backward <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(test)]

extern crate ndtensor;

use ndarray::Ix2;
use ndtensor::prelude::Tensor;

#[test]
fn test_backward() {
    let shape = (3, 3);

    let a = Tensor::<f64, Ix2>::linshape(shape.clone())
        .unwrap()
        .variable();
    let b = Tensor::<f64, Ix2>::ones(shape.clone()).variable();

    let res = &a + &b;

    let grad = res.into_dyn().grad().unwrap();

    assert_eq!(grad[a.id()], Tensor::<f64, Ix2>::ones(shape).into_dyn());
    assert_eq!(grad[b.id()], b.ones_like().into_dyn());
}

#[test]
fn test_mul() {
    let shape = (3, 3);

    let a = Tensor::<f64, Ix2>::linshape(shape.clone())
        .unwrap()
        .variable();
    let b = Tensor::<f64, Ix2>::ones(shape.clone()).variable();

    let res = &a * &b;

    let grad = res.into_dyn().grad().unwrap();

    assert_eq!(grad[a.id()], b.to_dyn());
    assert_eq!(grad[b.id()], a.to_dyn());
}

#[test]
fn test_sub() {
    let shape = (3, 3);

    let a = Tensor::<f64, Ix2>::linshape(shape.clone())
        .unwrap()
        .variable();
    let b = Tensor::<f64, Ix2>::ones(shape.clone()).variable();

    let res = &a - &b;

    let grad = res.into_dyn().grad().unwrap();

    assert_eq!(grad[a.id()], Tensor::<f64, Ix2>::ones(shape).into_dyn());
    assert_eq!(
        grad[b.id()],
        Tensor::<f64, Ix2>::ones(shape).into_dyn().neg()
    );
}

#[test]
fn test_div() {
    let shape = (3, 3);

    let a = Tensor::<f64, Ix2>::linshape(shape.clone())
        .unwrap()
        .variable();
    let b = Tensor::<f64, Ix2>::ones(shape.clone()).variable();

    let res = &a / &b;

    let grad = res.into_dyn().grad().unwrap();

    assert_eq!(grad[a.id()], b.to_dyn());
    assert_eq!(grad[b.id()], a.to_dyn().neg());
}

#[test]
fn test_pow() {
    let shape = (3, 3);

    let a = Tensor::<f64, Ix2>::linshape(shape.clone())
        .unwrap()
        .variable();

    let res = a.powi(2);

    let grad = res.into_dyn().grad().unwrap();
    let exp = a.mul_scalar(2f64).into_dyn();
    assert_eq!(grad[a.id()], exp);
}
