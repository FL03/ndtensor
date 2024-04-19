/*
    Appellation: ops <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::{Dimension, TensorExpr};
use crate::TensorBase;
use acme::prelude::{BinaryOp, Scalar, UnaryOp};
use nd::{ArrayBase, DimMax};
use nd::{Data, DataMut, DataOwned, OwnedRepr, RawDataClone, ScalarOperand};
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

macro_rules! scalar_op {
    ($(($method:ident, $variant:ident, $op:tt)),*) => {
        $(
            scalar_op!(@loop $method, $variant, $op);
        )*
    };

    (@loop $method:ident, $variant:ident, $op:tt) => {
        pub fn $method(&self, other: A) -> crate::Tensor<A, D> where A: ScalarOperand, S: Data<Elem = A> {
            let data = self.data() $op other;
            let op = TensorExpr::binary(
                self.to_owned().into_dyn().boxed(),
                crate::Tensor::from_scalar(other).into_dyn().boxed(),
                BinaryOp::$variant(),
            );
            new!(data, Some(op))
        }
    };

}

impl<A, S, D> TensorBase<S, D>
where
    A: Scalar,
    D: Dimension,
    S: Data<Elem = A> + DataOwned + RawDataClone,
{
    pub fn abs(&self) -> crate::Tensor<<A as Scalar>::Real, D>
    where
        A: Scalar<Real = A>,
    {
        let data = self.data().mapv(|x| x.abs());
        let op = TensorExpr::<S, S>::unary(self.clone().into_dyn().boxed(), UnaryOp::Abs);
        TensorBase::from_arr(data).with_op(op.into_owned())
    }

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
    binop!(
        (add, +),
        (div, /),
        (mul, *),
        (rem, %),
        (sub, -)
    );

    scalar_op!(
        (add_scalar, add, +),
        (div_scalar, div, /),
        (mul_scalar, mul, *),
        (rem_scalar, rem, %),
        (sub_scalar, sub, -)
    );

    unop!(cos, cosh, exp, ln, neg, sin, sinh, sqr, sqrt, tan, tanh);
}
impl<A, S, D> TensorBase<S, D>
where
    A: ComplexFloat,
    D: Dimension,
    S: Data<Elem = A> + DataOwned + RawDataClone,
{
    unop!(acos, acosh, asin, asinh, atan);
}

macro_rules! impl_unary_op {
    ($(($($path:ident)::*, $call:ident)),*) => {
        $(
            impl_unary_op!(@impl $($path)::*, $call);
        )*
    };
    ($($path:ident)::*, $call:ident) => {
        impl_unary_op!(@impl $($path)::*, $call);
    };
    (@impl $($path:ident)::*, $call:ident) => {
        impl<A, S, D> $($path)::* for TensorBase<S, D>
        where
            A: Clone + $($path)::*<Output = A>,
            D: Dimension,
            S: Data<Elem = A> + DataOwned + RawDataClone,
        {
            type Output = TensorBase<OwnedRepr<A>, D>;

            fn $call(self) -> Self::Output {
                let data = self.data().mapv(|x| x.$call());
                let op = TensorExpr::unary(
                    self.into_dyn().into_owned().boxed(),
                    UnaryOp::$call(),
                );
                TensorBase::from_arr(data).with_op(op)
            }
        }

        impl<'a, A, S, D> $($path)::* for &'a TensorBase<S, D>
        where
            A: Clone + $($path)::*<Output = A>,
            D: Dimension,
            S: Data<Elem = A> + DataOwned + RawDataClone,
        {
            type Output = TensorBase<OwnedRepr<A>, D>;

            fn $call(self) -> Self::Output {
                let data = self.data().mapv(|x| x.$call());
                let op = TensorExpr::unary(
                    self.to_owned().into_dyn().boxed(),
                    UnaryOp::$call(),
                );
                TensorBase::from_arr(data).with_op(op)
            }
        }
    };
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
                let op = TensorExpr::binary(
                    Box::new(lhs),
                    Box::new(TensorBase::from_arr(rhs.to_owned().into_dyn())),
                    BinaryOp::$target(),
                );
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
                let op = TensorExpr::binary(
                    Box::new(lhs),
                    Box::new(rhs.to_owned().into_dyn()),
                    BinaryOp::$target(),
                );
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
            A: Clone + core::ops::$bound<B, Output = A> + num::NumCast,
            B: Clone + num::ToPrimitive,
            D1: Dimension + DimMax<D2>,
            D2: Dimension,
            S1: DataOwned<Elem = A> + DataMut,
            S2: DataOwned<Elem = B>,

        {
            type Output = TensorBase<OwnedRepr<A>, <D1 as DimMax<D2>>::Output>;

            fn $call(self, rhs: TensorBase<S2, D2>) -> Self::Output {
                let data = core::ops::$bound::$call(self.data(), rhs.data());
                let op = unsafe { TensorExpr::binary(
                    Box::new(self.into_dyn().into_owned()),
                    Box::new(rhs.into_dyn().raw_view().cast::<A>().deref_into_view()),
                    BinaryOp::$call(),
                )};
                new!(data, Some(op.to_owned()))
            }
        }

        impl<'a, A, B, S1, S2, D1, D2> core::ops::$bound<TensorBase<S2, D2>> for &'a TensorBase<S1, D1>
        where
            A: Clone + core::ops::$bound<B, Output = A> + num::NumCast,
            B: Clone + num::ToPrimitive,
            D1: Dimension + DimMax<D2>,
            D2: Dimension,
            S1: DataOwned<Elem = A> + DataMut + RawDataClone,
            S2: DataOwned<Elem = B>,

        {
            type Output = TensorBase<OwnedRepr<A>, <D1 as DimMax<D2>>::Output>;

            fn $call(self, rhs: TensorBase<S2, D2>) -> Self::Output {
                let data = core::ops::$bound::$call(self.data(), rhs.data());
                let op = unsafe { TensorExpr::binary(
                    Box::new(self.to_owned().into_dyn()),
                    Box::new(rhs.into_dyn().raw_view().cast::<A>().deref_into_view()),
                    BinaryOp::$call(),
                )};
                new!(data, Some(op.to_owned()))
            }
        }

        impl<'a, A, B, S1, S2, D1, D2> core::ops::$bound<&'a TensorBase<S2, D2>> for &'a TensorBase<S1, D1>
        where
            A: Clone + core::ops::$bound<B, Output = A> + num::NumCast,
            B: Clone + num::ToPrimitive,
            D1: Dimension + DimMax<D2>,
            D2: Dimension,
            S1: DataOwned<Elem = A> + DataMut + RawDataClone,
            S2: DataOwned<Elem = B> + RawDataClone,

        {
            type Output = TensorBase<OwnedRepr<A>, <D1 as DimMax<D2>>::Output>;

            fn $call(self, rhs: &'a TensorBase<S2, D2>) -> Self::Output {
                let data = core::ops::$bound::$call(self.data(), rhs.data());
                let op = unsafe { TensorExpr::binary(
                    Box::new(self.to_owned().into_dyn()),
                    Box::new(rhs.to_dyn().raw_view().cast::<A>().deref_into_view()),
                    BinaryOp::$call(),
                )};
                new!(data, Some(op.to_owned()))
            }
        }

        impl<'a, A, B, S1, S2, D1, D2> core::ops::$bound<&'a TensorBase<S2, D2>> for TensorBase<S1, D1>
        where
            A: Clone + core::ops::$bound<B, Output = A> + num::NumCast,
            B: Clone + num::ToPrimitive,
            D1: Dimension + DimMax<D2>,
            D2: Dimension,
            S1: DataOwned<Elem = A> + DataMut + RawDataClone,
            S2: DataOwned<Elem = B> + RawDataClone,

        {
            type Output = TensorBase<OwnedRepr<A>, <D1 as DimMax<D2>>::Output>;

            fn $call(self, rhs: &'a TensorBase<S2, D2>) -> Self::Output {
                let data = core::ops::$bound::$call(self.data(), rhs.data());
                let op = unsafe { TensorExpr::binary(
                    Box::new(self.into_owned().into_dyn()),
                    Box::new(rhs.to_dyn().raw_view().cast::<A>().deref_into_view()),
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

impl_unary_op!((core::ops::Neg, neg), (core::ops::Not, not));
