/*
    Appellation: build <macros>
    Contrib: FL03 <jo3mccain@icloud.com>
*/

macro_rules! ndcreate {
    ($method:ident$(<$($t:ident),*>)?($($field:ident:$ty:ty),*) -> Result<$self:ty, $err:ty> $($rest:tt)*) => {
        pub fn $method$(<$($t),*>)?($($field:$ty),*) -> Result<$self, $err> $($rest)*
        {
            let arr = ArrayBase::$method($($field),*)?;
            Ok(new!(arr))
        }
    };
    ($method:ident$(<$($t:ident),*>)?($($field:ident:$ty:ty),*) -> $($rest:tt)*) => {
        pub fn $method$(<$($t),*>)?($($field:$ty),*) -> $($rest)* {
            new!(ArrayBase::$method($($field),*))
        }
    };

}

#[allow(unused_macros)]
macro_rules! ndbuilder {
    ($vis:vis $method:ident $($rest:tt)*) => {
        ndbuilder!(@fn $vis $method $($rest)*);
    };
    (@fn $vis:vis $method:ident<Sh>() -> $out:ty where $($rest:tt)*) => {
        ndbuilder!(@fn $vis $method.$method<Sh>() -> $out where $($rest)*);
    };
    (@fn $vis:vis $method:ident.$call:ident<Sh>() -> $out:ty where $($rest:tt)*) => {
        $vis fn $method<Sh>(shape: Sh)
        where
            Sh: ndarray::ShapeBuilder<Dim = D>,
            $($rest)*
        {
            let arr = ArrayBase::$method($($field),*)?;
            $crate::TensorBase::from_arr(arr)
        }
    };
}
