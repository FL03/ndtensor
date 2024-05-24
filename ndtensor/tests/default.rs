/*
    Appellation: default <test>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

pub fn multiply<A, B, C>(x: A, y: B) -> C
where
    A: core::ops::Mul<B, Output = C>,
{
    x * y
}

#[test]
fn compiles() {
    assert!(multiply(2, 3) > 0);
    assert_eq!(multiply(2, 3), 6);
    assert_ne!(multiply(2, 3), 7);
}
