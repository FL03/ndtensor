/*
    Appellation: ops <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use crate::prelude::{Tensor, TensorBase, TensorMode};
use nd::*;
use nd::linalg::Dot;
use num::complex::{Complex, ComplexFloat};

macro_rules! binop {
    ($(($method:ident, $op:tt)),*) => {
        $(
            binop!(@loop $method, $op);
        )*
    };

    (@loop $method:ident, $op:tt) => {
        pub fn $method(&self, other: &Self) -> Tensor<A, D, K> {
            let data = self.data() $op other.data();
            data.into()
        }
    };
}

macro_rules! unop {
    ($($method:ident),*) => {
        $(
            unop!(@loop $method);
        )*
    };
    (@loop $method:ident) => {
        pub fn $method(&self) -> Tensor<A, D, K> {
            let data = self.data().mapv(|x| x.$method());
            data.into()
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
        pub fn $method(&self, other: A) -> Tensor<A, D, K> {
            let data = self.data() $op other;
            data.into()
        }
    };

}

impl<A, S, D, K> TensorBase<S, D, K>
where
    A: ComplexFloat + LinalgScalar + ScalarOperand,
    D: Dimension,
    K: TensorMode,
    S: Data<Elem = A> + DataOwned + RawDataClone,
{
    pub fn abs(&self) -> Tensor<A::Real, D, K> {
        let data = self.data().mapv(|x| x.abs());
        // let op = TensorExpr::<S, S>::unary(self.clone().into_dyn().boxed(), UnaryOp::Abs);
        // TensorBase::from_arr(data).with_op(op.into_owned())
        TensorBase::from_arr(data)
    }

    pub fn matmul<B, T, E, U>(&self, rhs: &U) -> Tensor<B, E, K>
    where
        B: LinalgScalar,
        E: Dimension,
        ArrayBase<S, D>: Dot<U, Output = Array<B, E>>,
    {
        self.data().dot(&rhs).into()
    }

    pub fn powc(&self, n: Complex<A::Real>) -> Tensor<Complex<A::Real>, D, K> {
        self.mapv(|x| x.powc(n))
    }

    pub fn powi(&self, n: i32) -> Tensor<A, D, K> {
        self.mapv(|x| x.powi(n))
    }

    pub fn powf(&self, n: A::Real) -> Tensor<A, D, K> {
        self.mapv(|x| x.powf(n))
    }

    pub fn cubed(&self) -> Tensor<A, D, K> {
        self.powi(3)
    }

    pub fn sqrd(&self) -> Tensor<A, D, K> {
        self.powi(2)
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

    unop!(
        acos, acosh, asin, asinh, atan, cos, cosh, exp, ln, neg, recip, sin, sinh, sqrt, tan, tanh
    );
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
        impl<A, S, D, K> $($path)::* for TensorBase<S, D, K>
        where
            A: Clone + $($path)::*<Output = A>,
            D: Dimension,
            K: $crate::TensorMode,
            S: Data<Elem = A> + DataOwned + RawDataClone,
        {
            type Output = $crate::Tensor<A, D, K>;

            fn $call(self) -> Self::Output {
                self.mapv(|x| x.$call())
            }
        }

        impl<'a, A, S, D, K> $($path)::* for &'a TensorBase<S, D, K>
        where
            A: Clone + $($path)::*<Output = A>,
            D: Dimension,
            K: $crate::TensorMode,
            S: Data<Elem = A> + DataOwned + RawDataClone,
        {
            type Output = $crate::Tensor<A, D, K>;

            fn $call(self) -> Self::Output {
                self.mapv(|x| x.$call())
            }
        }
    };
}

macro_rules! impl_assign_op {
    ($($bound:ident.$call:ident),*) => {
        $(
            impl_assign_op!(@impl $bound.$call);
        )*
    };
    (@impl $bound:ident.$call:ident) => {
        impl<'a, A, S2, D1, D2, K> core::ops::$bound<&'a ArrayBase<S2, D2>> for $crate::Tensor<A, D1, K>
        where
            A: Clone + core::ops::$bound,
            D1: Dimension,
            D2: Dimension,
            K: $crate::TensorMode,
            S2: DataOwned<Elem = A> + RawDataClone,

        {
            fn $call(&mut self, rhs: &'a ArrayBase<S2, D2>) {

                // let op = TensorExpr::binary(
                //     self.to_dyn().boxed(),
                //     TensorBase::ndtensor(rhs.to_owned().into_dyn()).boxed(),
                //     BinaryOp::$call(),
                // );
                let mut data = self.data().clone();
                core::ops::$bound::$call(&mut data, rhs);
                *self = new!(data.clone())
            }
        }
        impl<'a, A, S2, D1, D2, K1, K2> core::ops::$bound<&'a TensorBase<S2, D2, K2>> for $crate::Tensor<A, D1, K1>
        where
            A: Clone + core::ops::$bound,
            D1: Dimension,
            D2: Dimension,
            K1: $crate::TensorMode,
            K2: $crate::TensorMode,
            S2: DataOwned<Elem = A> + RawDataClone,

        {
            fn $call(&mut self, rhs: &'a TensorBase<S2, D2, K2>) {
                // let op = TensorExpr::binary(
                //     self.to_dyn().boxed(),
                //     rhs.to_owned().into_dyn().boxed(),
                //     BinaryOp::$call(),
                // );
                let mut data = self.data().clone();
                core::ops::$bound::$call(&mut data, rhs.data());
                *self = new!(data.clone())
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
        impl_binary_op!(@arr $bound, $call);
        impl_binary_op!(@tensor $bound, $call);
    };
    (@arr $bound:ident, $call:ident) => {
        // impl_binary_op!(alt: $bound, $call, $op);

        impl<A, B, S1, S2, D1, D2, K> core::ops::$bound<ArrayBase<S2, D2>> for TensorBase<S1, D1, K>
        where
            ArrayBase<S1, D1>: core::ops::$bound<ArrayBase<S2, D2>, Output = ArrayBase<OwnedRepr<A>, <D1 as DimMax<D2>>::Output>>,
            A: Clone + num::NumCast,
            B: Clone + num::ToPrimitive,
            D1: Dimension + DimMax<D2>,
            D2: Dimension,
            K: $crate::TensorMode,
            S1: DataOwned<Elem = A> + DataMut,
            S2: DataOwned<Elem = B> + RawDataClone,

        {
            type Output = TensorBase<OwnedRepr<A>, <D1 as DimMax<D2>>::Output, K>;

            fn $call(self, rhs: ArrayBase<S2, D2>) -> Self::Output {
                // let op = TensorExpr::binary(
                //     self.to_owned().into_dyn().boxed(),
                //     TensorBase::ndtensor(rhs.mapv(|x| A::from(x).unwrap())).boxed(),
                //     BinaryOp::$call(),
                // );
                let data = core::ops::$bound::$call(self.data, rhs);

                new!(data)
            }
        }

        impl<'a, A, B, S1, S2, D1, D2, K> core::ops::$bound<&'a ArrayBase<S2, D2>> for TensorBase<S1, D1, K>
        where
            ArrayBase<S1, D1>: core::ops::$bound<&'a ArrayBase<S2, D2>, Output = ArrayBase<OwnedRepr<A>, <D1 as DimMax<D2>>::Output>>,
            A: Clone + num::NumCast,
            B: Clone + num::ToPrimitive,
            D1: Dimension + DimMax<D2>,
            D2: Dimension,
            K: $crate::TensorMode,
            S1: DataOwned<Elem = A> + DataMut,
            S2: DataOwned<Elem = B> + RawDataClone,

        {
            type Output = TensorBase<OwnedRepr<A>, <D1 as DimMax<D2>>::Output, K>;

            fn $call(self, rhs: &'a ArrayBase<S2, D2>) -> Self::Output {
                // let op = TensorExpr::binary(
                //     self.to_owned().into_dyn().boxed(),
                //     TensorBase::ndtensor(rhs.mapv(|x| A::from(x).unwrap())).boxed(),
                //     BinaryOp::$call(),
                // );
                let data = core::ops::$bound::$call(self.data, rhs);

                new!(data)
            }
        }

        impl<'a, A, B, S1, S2, D1, D2> core::ops::$bound<ArrayBase<S2, D2>> for &'a TensorBase<S1, D1>
        where
            &'a ArrayBase<S1, D1>: core::ops::$bound<ArrayBase<S2, D2>, Output = ArrayBase<OwnedRepr<A>, <D1 as DimMax<D2>>::Output>>,
            A: Clone + num::NumCast,
            B: Clone + num::ToPrimitive,
            D1: Dimension + DimMax<D2>,
            D2: Dimension,
            S1: DataOwned<Elem = A> + DataMut,
            S2: DataOwned<Elem = B> + RawDataClone,

        {
            type Output = TensorBase<OwnedRepr<A>, <D1 as DimMax<D2>>::Output>;

            fn $call(self, rhs: ArrayBase<S2, D2>) -> Self::Output {
                // let op = TensorExpr::binary(
                //     self.to_owned().into_dyn().boxed(),
                //     TensorBase::ndtensor(rhs.mapv(|x| A::from(x).unwrap())).boxed(),
                //     BinaryOp::$call(),
                // );
                let data = core::ops::$bound::$call(self.data(), rhs);

                new!(data)
            }
        }
    };
    (@tensor $bound:ident, $call:ident) => {
        impl<A, B, S1, S2, D1, D2, K1, K2> core::ops::$bound<TensorBase<S2, D2, K2>> for TensorBase<S1, D1, K1>
        where
            A: Clone + core::ops::$bound<B, Output = A> + num::NumCast,
            B: Clone + num::ToPrimitive,
            D1: Dimension + DimMax<D2>,
            D2: Dimension,
            K1: $crate::TensorMode,
            K2: $crate::TensorMode,
            S1: DataOwned<Elem = A> + DataMut,
            S2: DataOwned<Elem = B> + RawDataClone,

        {
            type Output = TensorBase<OwnedRepr<A>, <D1 as DimMax<D2>>::Output, $crate::Variable>;

            fn $call(self, rhs: TensorBase<S2, D2, K2>) -> Self::Output {
                let data = core::ops::$bound::$call(self.data(), rhs.data());
                // let op = TensorExpr::binary(
                //     Box::new(self.into_dyn().into_owned()),
                //     Box::new(rhs.numcast().into_dyn()),
                //     BinaryOp::$call(),
                // );
                new!(data)
            }
        }

        impl<'a, A, B, S1, S2, D1, D2, K1, K2> core::ops::$bound<TensorBase<S2, D2, K2>> for &'a TensorBase<S1, D1, K1>
        where
            A: Clone + core::ops::$bound<B, Output = A> + num::NumCast,
            B: Clone + num::ToPrimitive,
            D1: Dimension + DimMax<D2>,
            D2: Dimension,
            K1: $crate::TensorMode,
            K2: $crate::TensorMode,
            S1: DataOwned<Elem = A> + DataMut + RawDataClone,
            S2: DataOwned<Elem = B>,

        {
            type Output = TensorBase<OwnedRepr<A>, <D1 as DimMax<D2>>::Output, $crate::Variable>;

            fn $call(self, rhs: TensorBase<S2, D2, K2>) -> Self::Output {
                let data = core::ops::$bound::$call(self.data(), rhs.data());
                // let op = TensorExpr::binary(
                //     Box::new(self.to_owned().into_dyn()),
                //     Box::new(rhs.numcast().into_dyn()),
                //     BinaryOp::$call(),
                // );
                new!(data)
            }
        }

        impl<'a, A, B, S1, S2, D1, D2, K1, K2> core::ops::$bound<&'a TensorBase<S2, D2, K2>> for &'a TensorBase<S1, D1, K1>
        where
            A: Clone + core::ops::$bound<B, Output = A> + num::NumCast,
            B: Clone + num::ToPrimitive,
            D1: Dimension + DimMax<D2>,
            D2: Dimension,
            K1: $crate::TensorMode,
            K2: $crate::TensorMode,
            S1: DataOwned<Elem = A> + DataMut + RawDataClone,
            S2: DataOwned<Elem = B> + RawDataClone,

        {
            type Output = TensorBase<OwnedRepr<A>, <D1 as DimMax<D2>>::Output, $crate::Variable>;

            fn $call(self, rhs: &'a TensorBase<S2, D2, K2>) -> Self::Output {
                let data = core::ops::$bound::$call(self.data(), rhs.data());
                // let op = TensorExpr::binary(
                //     Box::new(self.to_owned().into_dyn()),
                //     Box::new(rhs.numcast().to_dyn()),
                //     BinaryOp::$call(),
                // );
                new!(data)
            }
        }

        impl<'a, A, B, S1, S2, D1, D2, K1, K2> core::ops::$bound<&'a TensorBase<S2, D2, K2>> for TensorBase<S1, D1, K1>
        where
            A: Clone + core::ops::$bound<B, Output = A> + num::NumCast,
            B: Clone + num::ToPrimitive,
            D1: Dimension + DimMax<D2>,
            D2: Dimension,
            K1: $crate::TensorMode,
            K2: $crate::TensorMode,
            S1: DataOwned<Elem = A> + DataMut + RawDataClone,
            S2: DataOwned<Elem = B> + RawDataClone,

        {
            type Output = TensorBase<OwnedRepr<A>, <D1 as DimMax<D2>>::Output, $crate::Variable>;

            fn $call(self, rhs: &'a TensorBase<S2, D2, K2>) -> Self::Output {
                let data = core::ops::$bound::$call(self.data(), rhs.data());
                // let op = TensorExpr::binary(
                //     Box::new(self.into_owned().into_dyn()),
                //     Box::new(rhs.to_dyn().numcast()),
                //     BinaryOp::$call(),
                // );
                new!(data)
            }
        }
    };
}

#[allow(unused_macros)]
macro_rules! impl_binop_scalar {
    ($t:ty: $($bound:ident.$call:ident),*) => {
        $(
            impl_binop_scalar!(@impl $bound.$call($t));
        )*
    };
    (@impl $bound:ident.$call:ident($t:ty)) => {
        // impl_binary_op!(alt: $bound, $call, $op);

        impl<A, S, D> core::ops::$bound<$t> for TensorBase<S, D, K>
        where
            A: Clone + num::NumCast,
            D: Dimension,
            K: $crate::TensorMode,
            S: DataOwned<Elem = A> + DataMut,
            ArrayBase<S, D>: core::ops::$bound<$t, Output = ArrayBase<OwnedRepr<$t>, D>>,
        {
            type Output = TensorBase<OwnedRepr<$t>, D, K>;

            fn $call(self, rhs: $t) -> Self::Output {
                // let op = TensorExpr::binary(
                //     self.numcast::<$t>().into_dyn().boxed(),
                //     TensorBase::from_scalar(rhs).into_owned().into_dyn().boxed(),
                //     BinaryOp::$call(),
                // );
                let data = core::ops::$bound::$call(self.data, rhs);

                new!(data)
            }
        }

        impl<'a, A, S, D, K> core::ops::$bound<$t> for &'a TensorBase<S, D, K>
        where
            A: Clone + num::NumCast,
            D: Dimension,
            S: DataOwned<Elem = A> + DataMut,
            K: $crate::TensorMode,
            ArrayBase<S, D>: core::ops::$bound<$t, Output = ArrayBase<OwnedRepr<$t>, D>>,
        {
            type Output = TensorBase<OwnedRepr<$t>, D, K>;

            fn $call(self, rhs: $t) -> Self::Output {
                // let op = TensorExpr::binary(
                //     self.to_owned().into_dyn().boxed(),
                //     TensorBase::from_scalar(rhs).into_owned().into_dyn().boxed(),
                //     BinaryOp::$call(),
                // );
                let data = core::ops::$bound::$call(self.data, rhs);

                new!(data)
            }
        }
    };
}

impl_assign_op!(
    AddAssign.add_assign,
    DivAssign.div_assign,
    MulAssign.mul_assign,
    RemAssign.rem_assign,
    SubAssign.sub_assign
);

impl_binary_op!((Add, add), (Div, div), (Mul, mul), (Rem, rem), (Sub, sub));

// impl_binop_scalar!(f32: Add.add, Div.div, Mul.mul, Sub.sub);
// impl_binop_scalar!(f64: Add.add, Div.div, Mul.mul, Sub.sub);

impl_unary_op!((core::ops::Neg, neg), (core::ops::Not, not));
