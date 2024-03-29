use rand::{distributions::uniform::SampleUniform};
use num_traits::{PrimInt, NumAssignOps, Num, NumCast};


/// All types that may be used in a [Tensor](crate::Tensor).
///
/// This trait gets implemented automatically for all types
/// that satisfy its parent traits.

pub trait Inner: PartialEq + Clone + Send + Sync + std::fmt::Debug {}
impl<T: PartialEq + Clone + Send + Sync + std::fmt::Debug> Inner for T {}


/// All numeric types.
///
/// This trait gets implemented automatically for all types
/// that satisfy its parent traits.

pub trait Numeric: Inner + Copy + PartialOrd + Num + NumCast + NumAssignOps + std::iter::Sum {}
impl<T: Inner + Copy + PartialOrd + Num + NumCast + NumAssignOps + std::iter::Sum> Numeric for T {}


/// All signed numeric types.
///
/// This trait gets implemented automatically for all types
/// that satisfy its parent traits.

pub trait Signed: Numeric + num_traits::Signed {}
impl<T: Numeric + num_traits::Signed> Signed for T {}


/// All unsigned numeric types.
///
/// This trait gets implemented automatically for all types
/// that satisfy its parent traits.

pub trait Unsigned: Numeric + num_traits::Unsigned {}
impl<T: Numeric + num_traits::Unsigned> Unsigned for T {}


/// All integer types.
///
/// This trait gets implemented automatically for all types
/// that satisfy its parent traits.

pub trait Integer: Numeric + PrimInt {}
impl<T: Numeric + PrimInt> Integer for T {}


/// All continuous numeric types.
///
/// This trait gets implemented automatically for all types
/// that satisfy its parent traits.

pub trait Real: Signed + num_traits::real::Real + SampleUniform + 'static {}
impl<T: Signed + num_traits::real::Real + SampleUniform + 'static> Real for T {}
