/*
    Appellation: inverse <error>
    Contrib: FL03 <jo3mccain@icloud.com>
*/
use strum::{AsRefStr, Display, EnumCount, EnumIs, EnumString, EnumIter, VariantArray, VariantNames};

#[derive(AsRefStr, Clone, Copy, Debug, Display, EnumCount, EnumIs, EnumIter, EnumString, Eq, Hash, Ord, PartialEq, PartialOrd, VariantArray, VariantNames)]
#[cfg_attr(feature = "serde", derive(serde::Deserialize, serde::Serialize,), serde(rename_all = "snake_case", untagged))]
#[repr(usize)]
#[strum(serialize_all = "snake_case")]
pub enum InverseError {
    NonSquareMatrix,
    SingularMatrix,
}

#[cfg(feature = "std")]
impl std::error::Error for InverseError {}