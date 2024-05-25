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

macro_rules! apply_view {
    ($call:ident$($rest:tt)*) => {
        apply_view!(@impl $call$($rest)*);
    };
    (@impl $call:ident(self) -> $out:ty where $($rest:tt)*) => {
        pub fn $call(self) -> $out where K: Copy, $($rest)* {
            apply_view!(@apply $call(self))
        }
    };
    (@impl $call:ident(&self) -> $out:ty where $($rest:tt)*) => {
        pub fn $call(&self) -> $out where K: Copy, $($rest)* {
            apply_view!(@apply $call(self))
        }
    };
    (@impl $call:ident(&mut self) -> $out:ty where $($rest:tt)*) => {
        pub fn $call(&mut self) -> $out where K: Copy, $($rest)* {
            apply_view!(@apply $call(self))
        }
    };
    (@apply $call:ident($self:expr)) => {
        $crate::TensorBase {
            id: $self.id,
            ctx: $self.ctx,
            data: $self.data.$call(),
        }
    };
}
