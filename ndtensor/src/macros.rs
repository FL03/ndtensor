/*
    Appellation: tensor <mod>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

macro_rules! new {
    {
        data:$data:expr,
        kind:$kind:expr,
        op:$op:expr,
    } => {
        $crate::tensor::TensorBase {
            id: $crate::prelude::TensorId::new(),
            ctx: $crate::Context::new($kind),

            data: $data,
            op: $op,
        }
    };
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

macro_rules! map_method {
    // ($method:ident) => {
    //     pub fn $method(&self) -> Self {
    //         new!(self.data.$method())
    //     }
    // };
    (a $method:ident$($rest:tt),*) => {
        map_method!(@impl $method$($rest)*);
    };
    ($method:ident($($field:ident:$ty:ty),*) where $($tb:ident: $($ext:ident)+),*) => {
        map_method!(@impl $method($($field:$ty),*) where $($tb: $($ext)+),*);
    };
    ($method:ident($($field:ident:$ty:ty),*) where $($tb:ident: $($ext:ident)+),* $($rest:tt),*) => {
        map_method!(@impl $method($($field:$ty),*) where $($tb: $($ext)+),*$($rest)*);
    };
    ($method:ident($($field:ident:$ty:ty),*) where $($tb:ident: $($ext:ident)+),* => $($res:ident),*) => {
        map_method!(@impl $method($($field:$ty),*) where $($tb: $($ext)+),* => $($res:ident),*);
    };
    ($method:ident<$($t:ident),*>($($field:ident:$ty:ty),*) where $($tb:ident: $($ext:ident)++),*) => {
        map_method!(@impl $method<$($t),*>($($field:$ty),*) where $($tb: $($ext)++),*);
    };
    (@impl $method:ident($($field:ident:$ty:ty),*) where $($tb:ident: $($ext:ident)+),* => $($res:ident),*) => {
        pub fn $method($($field:$ty),*) -> Result<$res, TensorError>
        where
            $($tb: $($ext)++),*
        {
            new!(ArrayBase::$method($($field),*)?)
        }
    };
    (@impl $method:ident($($field:ident:$ty:ty),*) where $($tb:ident: $($ext:ident)+),*) => {
        pub fn $method($($field:$ty),*) -> Self
        where
            $($tb: $($ext)++),*
        {
            new!(ArrayBase::$method($($field),*))
        }
    };
    (@impl $method:ident<$($t:ident),*>($($field:ident:$ty:ty),*) where $($tb:ident: $($ext:ident)++),*) => {
        pub fn $method<$($t),*>($($field:$ty),*) -> Self
        where
            $($tb: $($ext)++),*
        {
            new!(self.data.$method($($field),*))
        }
    };
}
