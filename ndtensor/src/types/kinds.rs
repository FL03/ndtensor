/*
    Appellation: mode <module>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use strum::{Display, EnumCount, EnumIs, EnumIter, EnumString, VariantNames};

pub trait TensorMode: Copy + 'static {
    const VARIABLE: bool;

    fn is<T: 'static>() -> bool {
        use core::any::TypeId;
        TypeId::of::<T>() == TypeId::of::<Variable>()
    }

    fn is_variable(&self) -> bool {
        Self::VARIABLE
    }
}

macro_rules! toggle {
    {type $T:ty, [$($name:ident($val:expr)),* $(,)?] $(,)?} => {
        $(
            toggle!(@impl $name<$T>: $val);
        )*
    };
    (@impl $name:ident<$T:ty>: $val:expr) => {
        #[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
        #[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize))]
        pub enum $name {}

        impl $name {
            pub const TOGGLE: $T = $val;

            pub fn is<T: 'static>() -> bool {
                use ::core::any::TypeId;
                TypeId::of::<T>() == TypeId::of::<Self>()
            }

            pub fn get(&self) -> $T {
                $val
            }
        }

        impl TensorMode for $name {
            const VARIABLE: bool = $val;
        }
    };

}

toggle! {
    type bool,
    [
        Normal(false),
        Variable(true)
    ]
}

#[derive(
    Clone,
    Copy,
    Debug,
    Default,
    Display,
    EnumCount,
    EnumIs,
    EnumIter,
    EnumString,
    Eq,
    Hash,
    Ord,
    PartialEq,
    PartialOrd,
    VariantNames,
)]
#[cfg_attr(
    feature = "serde",
    derive(serde::Deserialize, serde::Serialize,),
    serde(rename_all = "lowercase", untagged)
)]
#[repr(u8)]
#[strum(serialize_all = "lowercase")]
pub enum Mode {
    #[default]
    Normal = 0,
    Variable = 1,
}

impl Mode {
    pub fn new<K>() -> Self
    where
        K: 'static,
    {
        if Variable::is::<K>() {
            Self::Variable
        } else {
            Self::Normal
        }
    }

    pub fn from_bool(kind: bool) -> Self {
        if kind {
            Self::Variable
        } else {
            Self::Normal
        }
    }
    pub fn normal() -> Self {
        Self::Normal
    }

    pub fn variable() -> Self {
        Self::Variable
    }
}

impl From<Mode> for usize {
    fn from(mode: Mode) -> Self {
        mode as usize
    }
}

impl From<usize> for Mode {
    fn from(mode: usize) -> Self {
        match mode % Self::COUNT {
            0 => Self::Normal,
            _ => Self::Variable,
        }
    }
}

impl From<bool> for Mode {
    fn from(is_variable: bool) -> Self {
        Self::from_bool(is_variable)
    }
}
