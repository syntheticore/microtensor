#[cfg(not(feature = "threading"))]
use std::{
  rc::Rc,
  cell::{ Ref, RefMut, RefCell },
};

#[cfg(feature = "threading")]
use {
  std::sync::Arc,
  parking_lot::{ RwLock, RwLockReadGuard, RwLockWriteGuard },
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
pub type RefT<'a, T> = Ref<'a, T>;

#[cfg(feature = "threading")]
pub type RefT<'a, T> = RwLockReadGuard<'a, T>;


#[cfg(not(feature = "threading"))]
pub type RcCell<T> = Rc<RefCell<T>>;

#[cfg(feature = "threading")]
pub type RcCell<T> = Arc<RwLock<T>>;

#[cfg(not(feature = "threading"))]
pub type RefMutT<'a, T> = RefMut<'a, T>;

#[cfg(feature = "threading")]
pub type RefMutT<'a, T> = RwLockWriteGuard<'a, T>;


#[cfg(not(feature = "threading"))]
pub fn borrow<T>(value: &RcCell<T>) -> Ref<T> {
  value.borrow()
}

#[cfg(feature = "threading")]
pub fn borrow<T>(value: &RcCell<T>) -> RwLockReadGuard<T> {
  value.read()
}

#[cfg(not(feature = "threading"))]
pub fn borrow_mut<T>(value: &RcCell<T>) -> RefMut<T> {
  value.borrow_mut()
}

#[cfg(feature = "threading")]
pub fn borrow_mut<T>(value: &RcCell<T>) -> RwLockWriteGuard<T> {
  value.write()
}


pub(crate) fn make_rc_cell<T>(data: T) -> RcCell<T> {
  #[cfg(not(feature = "threading"))]
  return Rc::new(RefCell::new(data));

  #[cfg(feature = "threading")]
  return Arc::new(RwLock::new(data));
}


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
