/*
    Appellation: backward <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(test)]

extern crate ndtensor;

use ndarray::*;
use ndtensor::prelude::Tensor;

#[test]
fn test_backward() {
    let shape = (3, 3);

    let a = Tensor::<f64, Ix2>::linshape(shape.clone())
        .unwrap()
        .variable();
    let b = Tensor::<f64, Ix2>::ones(shape.clone()).variable();

    let res = &a + &b;

    let grad = res.grad().unwrap();

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

    let grad = res.grad().unwrap();

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

    let grad = res.grad().unwrap();

    assert_eq!(grad[a.id()], Tensor::<f64, Ix2>::ones(shape).into_dyn());
    assert_eq!(grad[b.id()], b.ones_like().into_dyn().neg());
}

#[test]
fn test_div() {
    let shape = (3, 3);

    let a = Tensor::<f64, Ix2>::linshape(shape.clone())
        .unwrap()
        .variable();
    let b = Tensor::<f64, Ix2>::ones(shape.clone()).variable();

    let res = a.div(&b);

    let grad = res.grad().unwrap();

    assert_eq!(grad[a.id()], b.to_dyn());
    assert_eq!(grad[b.id()], a.clone().neg().into_dyn());
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

#[test]
fn test_e() {
    let shape = (3, 3);

    let tensor = Tensor::<f64, Ix2>::linshape(shape.clone())
        .unwrap()
        .variable();
    let id = tensor.id();

    let ln = tensor.ln();
    let exp = tensor.exp();

    assert_eq!(ln.grad().unwrap()[id], tensor.recip().into_dyn());
    assert_eq!(exp.grad().unwrap()[id], tensor.exp().into_dyn());
}
#[test]
fn test_trig() {
    let shape = (3, 3);

    let tensor = Tensor::<f64, Ix2>::linshape(shape.clone())
        .unwrap()
        .variable();
    let id = tensor.id();

    let a = tensor.cos();
    let b = tensor.sin();
    let c = tensor.tan();
    assert_eq!(a.grad().unwrap()[id], -tensor.sin().into_dyn());
    assert_eq!(b.grad().unwrap()[id], tensor.cos().into_dyn());
    assert_eq!(
        c.grad().unwrap()[id],
        tensor.cos().powi(2).recip().into_dyn()
    );
}

#[test]
fn test_chained() {
    let shape = (3, 3);
    let dim = shape.clone().into_dimension();

    let a = Tensor::<f64, Ix2>::fill(dim.clone(), 2f64).variable();
    let b = Tensor::<f64, Ix2>::ones(dim.clone()).variable();

    let res = &a.mul(&a.add(&b));

    let grad = res.grad().unwrap();
    let exp = Tensor::fill(dim.clone(), 5f64).into_dyn();
    assert_eq!(grad[a.id()], exp);
    let exp = Tensor::fill(dim.clone(), 2f64).into_dyn();
    assert_eq!(grad[b.id()], exp);
}
