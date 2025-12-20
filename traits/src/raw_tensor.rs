/*
    Appellation: raw_tensor <module>
    Created At: 2025.12.20:06:50:01
    Contrib: @FL03
*/

pub trait RawSpace {
    type Elem: ?Sized;
}

/// A marker trait used to denote tensors that represent scalar values; more specifically, we
/// consider _**any**_ type implementing the [`RawTensorData`] type where the `Elem` associated
/// type is the implementor itself a scalar value.
pub trait ScalarTensorData: RawSpace<Elem = Self> {
    private! {}
}

/*
 ************* Implementations *************
*/

impl<T> ScalarTensorData for T
where
    T: RawSpace<Elem = Self>,
{
    seal! {}
}

macro_rules! impl_scalar_tensor {
    {$($T:ty),* $(,)?} => {
        $(
            impl RawSpace for $T {
                type Elem = $T;
            }
        )*
    };
}

impl_scalar_tensor! {
    u8, u16, u32, u64, u128, usize,
    i8, i16, i32, i64, i128, isize,
    f32, f64,
    bool, char, str
}

#[cfg(feature = "alloc")]
impl RawSpace for alloc::string::String {
    type Elem = u8;
}
