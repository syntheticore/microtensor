#[cfg(not(feature = "threading"))]
use std::{
  rc::Rc,
  cell::RefCell,
};

#[cfg(feature = "threading")]
use {
  std::sync::Arc,
  parking_lot::RwLock,
};

use rand::Rng;

use crate::{
  scalar::Real,
};


#[cfg(not(feature = "threading"))]
pub type RcT<T> = Rc<T>;

#[cfg(feature = "threading")]
pub type RcT<T> = Arc<T>;

#[cfg(not(feature = "threading"))]
pub type RcCell<T> = Rc<RefCell<T>>;

#[cfg(feature = "threading")]
pub type RcCell<T> = Arc<RwLock<T>>;


#[inline]
pub fn negative_index(i: isize, n: usize, start_behind: bool) -> usize {
  if i < 0 {
    let offset = if start_behind { 1 } else { 0 };
    let out = n as isize + i + offset;
    assert!(out >= 0, "Negative index {i} into rank {n} shape");
    out as usize
  } else {
    i as usize
  }
}


// Polar Box-Muller transformation

pub fn randn<T: Real>() -> (T, T) {
  let mut rng = rand::thread_rng();
  let u = rng.gen_range(-T::one(), T::one());
  let v = rng.gen_range(-T::one(), T::one());
  let r = u * u + v * v;
  // Try again if outside interval
  if r == T::zero() || r >= T::one() { return randn() }
  let c = (T::from(-2.0).unwrap() * r.ln() / r).sqrt();
  (u * c, v * c)
}
