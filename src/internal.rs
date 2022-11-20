use rand::Rng;

use crate::{
  scalar::Real,
};


#[inline]
pub fn negative_index(i: isize, n: usize, start_behind: bool) -> usize {
  if i < 0 {
    let offset = if start_behind { 1 } else { 0 };
    (n as isize + i + offset) as usize
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
