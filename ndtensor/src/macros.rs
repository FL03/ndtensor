/*
    Appellation: tensor <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

#[macro_use]
mod build;
#[macro_use]
mod create;
#[macro_use]
mod view;

macro_rules! new {
    ($data:expr) => {
        $crate::TensorBase::new($data)
    };
    ($data:expr, $op:expr) => {
        $crate::TensorBase::new($data)
    };
    ($data:expr, $op:expr, $kind:expr) => {
        $crate::TensorBase::new($data)
    };
}

macro_rules! fwd_expr_call {
    ($self:ident.$($rest:tt)*) => {
        match $self {
            $crate::grad::ops::TensorExpr::Binary { lhs, rhs, op } => $crate::grad::ops::TensorExpr::Binary {
                lhs: lhs.$($rest)*,
                rhs: rhs.$($rest)*,
                op,
            },
            $crate::grad::ops::TensorExpr::Unary { recv, op } => $crate::grad::ops::TensorExpr::Unary {
                recv: recv.$($rest)*,
                op,
            },
            $crate::grad::ops::TensorExpr::Matmul { lhs, rhs } => $crate::grad::ops::TensorExpr::Matmul {
                lhs: lhs.$($rest)*,
                rhs: rhs.$($rest)*,
            },
            $crate::grad::ops::TensorExpr::Transpose(recv) => $crate::grad::ops::TensorExpr::Transpose(recv.$($rest)*),
        }
    };
    (&$self:ident.$($rest:tt)*) => {
        match $self {
            $crate::grad::ops::TensorExpr::Binary { lhs, rhs, op } => $crate::grad::ops::TensorExpr::Binary {
                lhs: lhs.$($rest)*,
                rhs: rhs.$($rest)*,
                op: *op,
            },
            $crate::grad::ops::TensorExpr::Unary { recv, op } => $crate::grad::ops::TensorExpr::Unary {
                recv: recv.$($rest)*,
                op: *op,
            },
            $crate::grad::ops::TensorExpr::Matmul { lhs, rhs } => $crate::grad::ops::TensorExpr::Matmul {
                lhs: lhs.$($rest)*,
                rhs: rhs.$($rest)*,
            },
            $crate::grad::ops::TensorExpr::Transpose(recv) => $crate::grad::ops::TensorExpr::Transpose(recv.$($rest)*),
        }
    };
}

macro_rules! fwd_view_body {
    ($self:ident, $method:ident) => {
        fwd_expr_call!($self.$method().boxed())
    };
    (&$self:ident, $method:ident) => {
        fwd_expr_call!(&$self.as_ref().$method().boxed())
    };
    (&mut $self:ident, $method:ident) => {
        fwd_expr_call!(&$self.as_mut().$method().boxed())
    };
}
