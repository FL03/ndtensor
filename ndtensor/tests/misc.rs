/*
    Appellation: misc <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
#![cfg(test)]

extern crate ndtensor;

use lazy_static::lazy_static;
use ndarray::IntoDimension;
use ndtensor::prelude::hash_dim;

lazy_static! {
    static ref DIM: (usize, usize) = (3, 3);
    static ref DIM_HASH: u64 = 1069660947015105383;
}

#[test]
fn test_dim_hash() {
    // hash: 1069660947015105383
    let dim = (3, 3).into_dimension();
    let s1 = (3, 3);
    let s2 = [3, 3];
    let s3 = vec![3, 3];
    assert_eq!(hash_dim(dim), *DIM_HASH);
    assert_eq!(hash_dim(dim), hash_dim(s1));
    assert_eq!(hash_dim(dim), hash_dim(s2));
    assert_eq!(hash_dim(dim), hash_dim(s3));
}
