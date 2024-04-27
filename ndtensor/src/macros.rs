/*
    Appellation: tensor <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

macro_rules! new {
    ($data:expr) => {
        $crate::TensorBase::new($data, None, false)
    };
    ($data:expr, $op:expr) => {
        $crate::TensorBase::new($data, $op, false)
    };
    ($data:expr, $op:expr, $kind:expr) => {
        $crate::TensorBase::new($data, Some($op), $kind)
    };
}

macro_rules! create {
    ($data:expr, $($rest:tt)*) => {
        create!(@base $data, $($rest)*)
    };
    (@base $data:expr,) => {
        $crate::TensorBase {
            id: $crate::TensorId::new(),
            ctx: $crate::Context::new(false, $data.ndim()),
            data: $data,
            op: $crate::ops::TensorOp::none(),
        }
    };
    (@base $data:expr, op:$op:expr) => {
        $crate::TensorBase {
            id: $crate::TensorId::new(),
            ctx: $crate::Context::new(false, $data.ndim()),
            data: $data,
            op: $op,
        }
    };
    (@base $data:expr, kind:$kind:expr, op:$op:expr) => {
        $crate::TensorBase {
            id: $crate::TensorId::new(),
            ctx: $crate::Context::new($kind, $data.ndim()),
            data: $data,
            op: $op,
        }
    };
    (@base $data:expr, id:$id:expr, ctx:$ctx:expr, op:$op:expr) => {
        $crate::TensorBase {
            id: $id,
            ctx: $ctx,
            data: $data,
            op: $op,
        }
    };
}

macro_rules! fwd_expr_call {
    ($self:ident.$($rest:tt)*) => {
        match $self {
            $crate::ops::TensorExpr::Binary { lhs, rhs, op } => $crate::ops::TensorExpr::Binary {
                lhs: lhs.$($rest)*,
                rhs: rhs.$($rest)*,
                op,
            },
            $crate::ops::TensorExpr::Unary { recv, op } => $crate::ops::TensorExpr::Unary {
                recv: recv.$($rest)*,
                op,
            },
            $crate::ops::TensorExpr::Matmul { lhs, rhs } => $crate::ops::TensorExpr::Matmul {
                lhs: lhs.$($rest)*,
                rhs: rhs.$($rest)*,
            },
            $crate::ops::TensorExpr::Transpose(recv) => $crate::ops::TensorExpr::Transpose(recv.$($rest)*),
        }
    };
    (&$self:ident.$($rest:tt)*) => {
        match $self {
            $crate::ops::TensorExpr::Binary { lhs, rhs, op } => $crate::ops::TensorExpr::Binary {
                lhs: lhs.$($rest)*,
                rhs: rhs.$($rest)*,
                op: *op,
            },
            $crate::ops::TensorExpr::Unary { recv, op } => $crate::ops::TensorExpr::Unary {
                recv: recv.$($rest)*,
                op: *op,
            },
            $crate::ops::TensorExpr::Matmul { lhs, rhs } => $crate::ops::TensorExpr::Matmul {
                lhs: lhs.$($rest)*,
                rhs: rhs.$($rest)*,
            },
            $crate::ops::TensorExpr::Transpose(recv) => $crate::ops::TensorExpr::Transpose(recv.$($rest)*),
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

macro_rules! ndcreate {
    ($method:ident$(<$($t:ident),*>)?($($field:ident:$ty:ty),*) -> Result<$self:ty, $err:ty> $($rest:tt)*) => {
        pub fn $method$(<$($t),*>)?($($field:$ty),*) -> Result<$self, $err> $($rest)*
        {
            let arr = ArrayBase::$method($($field),*)?;
            Ok(new!(arr))
        }
    };
    ($method:ident$(<$($t:ident),*>)?($($field:ident:$ty:ty),*) -> $($rest:tt)*) => {
        pub fn $method$(<$($t),*>)?($($field:$ty),*) -> $($rest)*
        {
            new!(ArrayBase::$method($($field),*))
        }
    };

}

macro_rules! apply_view {
    ($call:ident$($rest:tt)*) => {
        apply_view!(@impl $call$($rest)*);
    };
    (@impl $call:ident(self) -> $out:ty where $($rest:tt)*) => {
        pub fn $call(self) -> $out where $($rest)* {
            apply_view!(@apply $call(self))
        }
    };
    (@impl $call:ident(&self) -> $out:ty where $($rest:tt)*) => {
        pub fn $call(&self) -> $out where $($rest)* {
            apply_view!(@apply $call(self))
        }
    };
    (@impl $call:ident(&mut self) -> $out:ty where $($rest:tt)*) => {
        pub fn $call(&mut self) -> $out where $($rest)* {
            apply_view!(@apply $call(self))
        }
    };
    (@apply $call:ident($self:expr)) => {
        $crate::TensorBase {
            id: $self.id,
            ctx: $self.ctx,
            data: $self.data.$call(),
            op: $self.op.$call(),
        }
    };
}
