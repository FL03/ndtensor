/*
    appellation: impl_tensor_ops <module>
    authors: @FL03
*/
use crate::tensor::TensorBase;

use crate::{Inverse, Tensor, TensorView, Transpose};

use ndarray::linalg::Dot;
use ndarray::{
    Array, ArrayBase, Data, DimMax, Dimension, Ix2, LinalgScalar, RawData, ScalarOperand,
};
use num_traits::NumAssign;

impl<A, S, D> TensorBase<S, D>
where
    D: Dimension,
    S: RawData<Elem = A>,
{
    /// transpose the current tensor, returning a new instance with a view of the transposed data.
    pub fn transpose(&self) -> TensorView<'_, A, D>
    where
        S: Data,
    {
        TensorBase {
            store: self.store().t(),
        }
    }
}

impl<A> Inverse for Tensor<A, Ix2>
where
    A: Copy + NumAssign + ScalarOperand,
{
    type Output = Tensor<A, Ix2>;

    fn inverse(&self) -> Self::Output {
        let store = self.store().inverse().expect("Matrix is not invertible");
        TensorBase { store }
    }
}

impl<'a, A, S, D> Transpose for &'a TensorBase<S, D>
where
    A: 'a,
    S: Data<Elem = A>,
    D: Dimension,
{
    type Output = TensorView<'a, A, D>;

    fn transpose(&self) -> Self::Output {
        let store = self.store().t();
        TensorBase { store }
    }
}

impl<A, S, D, X, S2, D2> Dot<X> for TensorBase<S, D>
where
    A: LinalgScalar,
    D: Dimension,
    D2: Dimension,
    S: RawData<Elem = A>,
    S2: RawData<Elem = A>,
    ArrayBase<S, D>: Dot<X, Output = ArrayBase<S2, D2>>,
{
    type Output = TensorBase<S2, D2>;

    fn dot(&self, rhs: &X) -> Self::Output {
        self.mapd(|store| Dot::dot(store, rhs))
    }
}

macro_rules! impl_unary_op {
    (@impl $trait:ident::$method:ident) => {
        impl<A, B, S, D> ::core::ops::$trait for TensorBase<S, D>
        where
            A: ScalarOperand + ::core::ops::$trait<Output = B>,
            D: Dimension,
            S: RawData<Elem = A>,
            ArrayBase<S, D>: ::core::ops::$trait<Output = Array<B, D>>,
        {
            type Output = Tensor<B, D>;

            fn $method(self) -> Self::Output {
                TensorBase {
                    store: ::core::ops::$trait::$method(self.store),
                }
            }
        }

        impl<'a, A, B, S, D> ::core::ops::$trait for &'a TensorBase<S, D>
        where
            A: ScalarOperand + ::core::ops::$trait<Output = B>,
            D: Dimension,
            S: RawData<Elem = A>,
            &'a ArrayBase<S, D>: ::core::ops::$trait<Output = Array<B, D>>,
        {
            type Output = Tensor<B, D>;

            fn $method(self) -> Self::Output {
                TensorBase {
                    store: ::core::ops::$trait::$method(self.store()),
                }
            }
        }

        impl<'a, A, B, S, D> ::core::ops::$trait for &'a mut TensorBase<S, D>
        where
            A: ScalarOperand + ::core::ops::$trait<Output = B>,
            D: Dimension,
            S: RawData<Elem = A>,
            &'a mut ArrayBase<S, D>: ::core::ops::$trait<Output = Array<B, D>>,
        {
            type Output = Tensor<B, D>;

            fn $method(self) -> Self::Output {
                TensorBase {
                    store: ::core::ops::$trait::$method(self.store_mut()),
                }
            }
        }
    };
    ($(
        $trait:ident::$method:ident
    ),* $(,)?) => {
        $(
            impl_unary_op!(@impl $trait::$method);
        )*
    };
}

macro_rules! impl_binary {
    (@impl $trait:ident::$method:ident) => {
        impl<A, B, C, S, D, S2, D2> ::core::ops::$trait<TensorBase<S2, D2>> for TensorBase<S, D>
        where
            A: ScalarOperand + ::core::ops::$trait<B, Output = C>,
            D: Dimension + DimMax<D2>,
            D2: Dimension,
            S: Data<Elem = A>,
            S2: Data<Elem = B>,
            ArrayBase<S, D>: ::core::ops::$trait<ArrayBase<S2, D2>, Output = Array<C, <D as DimMax<D2>>::Output>>,
        {
            type Output = Tensor<C, <D as DimMax<D2>>::Output>;

            fn $method(self, rhs: TensorBase<S2, D2>) -> Tensor<C, <D as DimMax<D2>>::Output> {
                TensorBase {
                    store: ::core::ops::$trait::$method(self.store, rhs.store),
                }
            }
        }

        impl<'a, A, S, D, B, S2, D2, C> ::core::ops::$trait<&'a TensorBase<S2, D2>> for TensorBase<S, D>
        where
            A: ScalarOperand + ::core::ops::$trait<B, Output = C>,
            D: Dimension + DimMax<D2>,
            D2: Dimension,
            S: Data<Elem = A>,
            S2: Data<Elem = B>,
            ArrayBase<S, D>: ::core::ops::$trait<&'a ArrayBase<S2, D2>, Output = Array<C, <D as DimMax<D2>>::Output>>,
        {
            type Output = Tensor<C, <D as DimMax<D2>>::Output>;

            fn $method(self, rhs: &'a TensorBase<S2, D2>) -> Tensor<C, <D as DimMax<D2>>::Output> {
                TensorBase {
                    store: ::core::ops::$trait::$method(self.store, rhs.store()),
                }
            }
        }

        impl<'a, A, S, D, B, S2, D2, C> ::core::ops::$trait<&'a mut TensorBase<S2, D2>> for TensorBase<S, D>
        where
            A: ScalarOperand + ::core::ops::$trait<B, Output = C>,
            D: Dimension + DimMax<D2>,
            D2: Dimension,
            S: Data<Elem = A>,
            S2: Data<Elem = B>,
            ArrayBase<S, D>: ::core::ops::$trait<&'a mut ArrayBase<S2, D2>, Output = Array<C, <D as DimMax<D2>>::Output>>,
        {
            type Output = Tensor<C, <D as DimMax<D2>>::Output>;

            fn $method(self, rhs: &'a mut TensorBase<S2, D2>) -> Tensor<C, <D as DimMax<D2>>::Output> {
                TensorBase {
                    store: ::core::ops::$trait::$method(self.store, rhs.store_mut()),
                }
            }
        }

        impl<'a, A, S, D, B, S2, D2, C> ::core::ops::$trait<&'a TensorBase<S2, D2>> for &'a TensorBase<S, D>
        where
            A: ScalarOperand + ::core::ops::$trait<B, Output = C>,
            D: Dimension + DimMax<D2>,
            D2: Dimension,
            S: Data<Elem = A>,
            S2: Data<Elem = B>,
            &'a ArrayBase<S, D>: ::core::ops::$trait<&'a ArrayBase<S2, D2>, Output = Array<C, <D as DimMax<D2>>::Output>>,
        {
            type Output = Tensor<C, <D as DimMax<D2>>::Output>;

            fn $method(self, rhs: &'a TensorBase<S2, D2>) -> Tensor<C, <D as DimMax<D2>>::Output> {
                TensorBase {
                    store: ::core::ops::$trait::$method(self.store(), rhs.store()),
                }
            }
        }

        impl<'a, A, S, D, B, S2, D2, C> ::core::ops::$trait<TensorBase<S2, D2>> for &'a TensorBase<S, D>
        where
            A: ScalarOperand + ::core::ops::$trait<B, Output = C>,
            D: Dimension + DimMax<D2>,
            D2: Dimension,
            S: Data<Elem = A>,
            S2: Data<Elem = B>,
            &'a ArrayBase<S, D>: ::core::ops::$trait<ArrayBase<S2, D2>, Output = Array<C, <D as DimMax<D2>>::Output>>,
        {
            type Output = Tensor<C, <D as DimMax<D2>>::Output>;

            fn $method(self, rhs: TensorBase<S2, D2>) -> Tensor<C, <D as DimMax<D2>>::Output> {
                TensorBase {
                    store: ::core::ops::$trait::$method(self.store(), rhs.store),
                }
            }
        }

        impl<'a, A, S, D, B, S2, D2, C> ::core::ops::$trait<TensorBase<S2, D2>> for &'a mut TensorBase<S, D>
        where
            A: ScalarOperand + ::core::ops::$trait<B, Output = C>,
            D: Dimension + DimMax<D2>,
            D2: Dimension,
            S: Data<Elem = A>,
            S2: Data<Elem = B>,
            &'a mut ArrayBase<S, D>: ::core::ops::$trait<ArrayBase<S2, D2>, Output = Array<C, <D as DimMax<D2>>::Output>>,
        {
            type Output = Tensor<C, <D as DimMax<D2>>::Output>;

            fn $method(self, rhs: TensorBase<S2, D2>) -> Tensor<C, <D as DimMax<D2>>::Output> {
                TensorBase {
                    store: ::core::ops::$trait::$method(self.store_mut(), rhs.store),
                }
            }
        }
    };
    ($(
        $trait:ident::$method:ident
    ),* $(,)?) => {
        paste::paste! {

            $(
                impl_binary!(@impl $trait::$method);
                impl_binary_assign!(@impl [<$trait Assign>]::[<$method _assign>]);
            )*
        }
    };
}

macro_rules! impl_binary_assign {
    (@impl $trait:ident::$method:ident) => {
        impl<A, B, S, D, S2, D2> ::core::ops::$trait<TensorBase<S2, D2>> for TensorBase<S, D>
        where
            A: ScalarOperand + ::core::ops::$trait<B>,
            D: Dimension + DimMax<D2>,
            D2: Dimension,
            S: Data<Elem = A>,
            S2: Data<Elem = B>,
            ArrayBase<S, D>: ::core::ops::$trait<ArrayBase<S2, D2>>,
        {

            fn $method(&mut self, rhs: TensorBase<S2, D2>) {
                ::core::ops::$trait::$method(self.store_mut(), rhs.store)
            }
        }

        impl<'a, A, B, S, D, S2, D2> ::core::ops::$trait<&'a TensorBase<S2, D2>> for TensorBase<S, D>
        where
            A: ScalarOperand + ::core::ops::$trait<B>,
            D: Dimension + DimMax<D2>,
            D2: Dimension,
            S: Data<Elem = A>,
            S2: Data<Elem = B>,
            ArrayBase<S, D>: ::core::ops::$trait<&'a ArrayBase<S2, D2>>,
        {

            fn $method(&mut self, rhs: &'a TensorBase<S2, D2>) {
                ::core::ops::$trait::$method(self.store_mut(), rhs.store())
            }
        }
    };
    ($(
        $trait:ident::$method:ident
    ),* $(,)?) => {
        $(
            impl_binary!(@impl $trait::$method);
        )*
    };
}

impl_binary! {
    Add::add,
    Div::div,
    Mul::mul,
    Rem::rem,
    Sub::sub,
}

impl_unary_op! {
    Neg::neg,
    Not::not,
}
