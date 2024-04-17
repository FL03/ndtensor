/*
    Appellation: ops <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::{Dimension, TensorExpr};
use crate::TensorBase;
use acme::prelude::{BinaryOp, Scalar, UnaryOp};
use ndarray::{ArrayBase, DimMax};
use ndarray::{Data, DataMut, DataOwned, OwnedRepr, RawDataClone};
use num::complex::ComplexFloat;

macro_rules! unop {
    ($($method:ident),*) => {
        $(
            unop!(@loop $method);
        )*
    };
    (@loop $method:ident) => {
        pub fn $method(&self) -> crate::Tensor<A, D> {
            let data = self.data().mapv(|x| x.$method());
            let op = TensorExpr::Unary {
                recv: Box::new(self.clone().into_dyn().into_owned()),
                op: UnaryOp::$method(),
            };
            new!(data, Some(op))
        }
    };
}

macro_rules! binop {
    ($(($method:ident, $op:tt)),*) => {
        $(
            binop!(@loop $method, $op);
        )*
    };
    (@loop $method:ident, $op:tt) => {
        pub fn $method(&self, other: &Self) -> crate::Tensor<A, D> {
            let data = self.data() $op other.data();
            let op = TensorExpr::binary(
                self.clone().into_dyn().boxed(),
                other.clone().into_dyn().boxed(),
                BinaryOp::$method(),
            );
            new!(data, Some(op.into_owned()))
        }
    };

}
impl<A, S, D> TensorBase<S, D>
where
    A: Scalar,
    D: Dimension,
    S: Data<Elem = A> + DataOwned + RawDataClone,
{
    pub fn powi(&self, n: i32) -> crate::Tensor<A, D> {
        let data = self.data().mapv(|x| x.powi(n));
        let op = TensorExpr::<S, S>::binary(
            self.clone().into_dyn().boxed(),
            TensorBase::from_scalar(A::from(n).unwrap())
                .into_dyn()
                .boxed(),
            BinaryOp::pow(),
        );
        TensorBase::from_arr(data).with_op(op.into_owned())
    }

    pub fn powf(&self, n: <A as Scalar>::Real) -> crate::Tensor<A, D> {
        let data = self.data().mapv(|x| x.powf(n));
        let op = TensorExpr::<S, S>::binary(
            self.clone().into_dyn().boxed(),
            TensorBase::from_scalar(A::from(n).unwrap())
                .into_dyn()
                .boxed(),
            BinaryOp::pow(),
        );
        TensorBase::from_arr(data).with_op(op.into_owned())
    }
}
impl<A, S, D> TensorBase<S, D>
where
    A: ComplexFloat,
    D: Dimension,
    S: Data<Elem = A> + DataOwned + RawDataClone,
{
    pub fn abs(&self) -> crate::Tensor<<A as ComplexFloat>::Real, D>
    where
        A: ComplexFloat<Real = A>,
    {
        let data = self.data().mapv(|x| x.abs());
        let op = TensorExpr::<S, S>::unary(self.clone().into_dyn().boxed(), UnaryOp::Abs);
        TensorBase::from_arr(data).with_op(op.into_owned())
    }
    binop!(
        (add, +),
        (div, /),
        (mul, *),
        (rem, %),
        (sub, -)
    );
    unop!(acos, acosh, asin, asinh, atan, cos, cosh, exp, ln, neg, sin, sinh, sqrt, tan, tanh);
}
macro_rules! impl_assign_op {
    ($(($bound:ident, $target:ident, $call:ident)),*) => {
        $(
            impl_assign_op!($bound, $target, $call);
        )*
    };
    ($bound:ident, $target:ident, $call:ident) => {
        impl<'a, A, S2, D1, D2> core::ops::$bound<&'a ArrayBase<S2, D2>> for $crate::Tensor<A, D1>
        where
            A: Clone + core::ops::$bound,
            D1: Dimension,
            D2: Dimension,
            S2: DataOwned<Elem = A> + RawDataClone,

        {

            fn $call(&mut self, rhs: &'a ArrayBase<S2, D2>) {

                let lhs = self.to_dyn();
                let op = { TensorExpr::binary(
                    Box::new(lhs),
                    Box::new(TensorBase::from_arr(rhs.to_owned().into_dyn())),
                    BinaryOp::$target(),
                )};
                let mut data = self.data().clone();
                core::ops::$bound::$call(&mut data, rhs);
                *self = new!(data.clone(), Some(op))
            }
        }
        impl<'a, A, S2, D1, D2> core::ops::$bound<&'a TensorBase<S2, D2>> for $crate::Tensor<A, D1>
        where
            A: Clone + core::ops::$bound,
            D1: Dimension,
            D2: Dimension,
            S2: DataOwned<Elem = A> + RawDataClone,

        {

            fn $call(&mut self, rhs: &'a TensorBase<S2, D2>) {

                let lhs = self.to_dyn();
                let op = { TensorExpr::binary(
                    Box::new(lhs),
                    Box::new(rhs.to_owned().into_dyn()),
                    BinaryOp::$target(),
                )};
                let mut data = self.data().clone();
                core::ops::$bound::$call(&mut data, rhs.data());
                *self = new!(data.clone(), Some(op))
            }
        }
    };
}

macro_rules! impl_binary_op {
    ($(($bound:ident, $call:ident)),*) => {
        $(
            impl_binary_op!($bound, $call);
        )*
    };
    ($bound:ident, $call:ident) => {
        // impl_binary_op!(alt: $bound, $call, $op);

        impl<A, B, S1, S2, D1, D2> core::ops::$bound<TensorBase<S2, D2>> for TensorBase<S1, D1>
        where
            A: Clone + core::ops::$bound<B, Output = A>,
            B: Clone,
            D1: Dimension + DimMax<D2>,
            D2: Dimension,
            S1: DataOwned<Elem = A> + DataMut,
            S2: DataOwned<Elem = B>,

        {
            type Output = TensorBase<OwnedRepr<A>, <D1 as DimMax<D2>>::Output>;

            fn $call(self, rhs: TensorBase<S2, D2>) -> Self::Output {
                let data = core::ops::$bound::$call(self.data(), rhs.data());
                let lhs = self.into_dyn().into_owned();
                let op = unsafe { TensorExpr::binary(
                    Box::new(lhs),
                    Box::new(rhs.into_dyn().raw_view().cast::<A>().deref_into_view()),
                    BinaryOp::$call(),
                )};
                new!(data, Some(op.to_owned()))
            }
        }

        impl<'a, A, B, S1, S2, D1, D2> core::ops::$bound<TensorBase<S2, D2>> for &'a TensorBase<S1, D1>
        where
            A: Clone + core::ops::$bound<B, Output = A>,
            B: Clone,
            D1: Dimension + DimMax<D2>,
            D2: Dimension,
            S1: DataOwned<Elem = A> + DataMut + RawDataClone,
            S2: DataOwned<Elem = B>,

        {
            type Output = TensorBase<OwnedRepr<A>, <D1 as DimMax<D2>>::Output>;

            fn $call(self, rhs: TensorBase<S2, D2>) -> Self::Output {
                let data = core::ops::$bound::$call(self.data(), rhs.data());
                let lhs = self.clone().into_dyn().into_owned();
                let op = unsafe { TensorExpr::binary(
                    Box::new(lhs),
                    Box::new(rhs.into_dyn().raw_view().cast::<A>().deref_into_view()),
                    BinaryOp::$call(),
                )};
                new!(data, Some(op.to_owned()))
            }
        }
    };
    (alt: $bound:ident, $call:ident, $op:tt) => {


        impl<A, B, C, S, D> core::ops::$bound<B> for TensorBase<S, D>
        where
            A: Clone + core::ops::$bound<B, Output = C>,
            D: Dimension + DimMax<D2>,
            S: DataOwned<Elem = A> + DataMut,
            ArrayBase<S, D>: core::ops::$bound<B, Output = C>,

        {
            type Output = C;

            fn $call(self, rhs: B) -> Self::Output {
                let data = core::ops::$bound::$call(self.data(), rhs.data());
                let lhs = self.into_dyn().into_owned();
                let op = unsafe { TensorExpr::binary(
                    Box::new(lhs),
                    Box::new(rhs.into_dyn().raw_view().cast::<A>().deref_into_view()),
                    BinaryOp::$call(),
                )};
                new!(data, Some(op.to_owned()))
            }
        }
    };
}

impl_binary_op!((Add, add), (Div, div), (Mul, mul), (Rem, rem), (Sub, sub));

impl_assign_op!(
    (AddAssign, add, add_assign),
    (DivAssign, div, div_assign),
    (MulAssign, mul, mul_assign),
    (RemAssign, rem, rem_assign),
    (SubAssign, sub, sub_assign)
);
