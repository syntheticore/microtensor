#[cfg(not(feature = "threading"))]
use std::{
  rc::Rc,
};

use std::fmt::Debug;

use rand::{ Rng, prelude::SliceRandom, distributions::{ Distribution, WeightedIndex }};
use num_traits::NumCast;
use serde::{ Serialize, Deserialize };

mod cops;
mod lops;

use crate::{
  internal::*,
  shape::{ Shape, DimensionIterator, SwitchIterator },
  variable::{ Variable, Traintape },
  scalar::{ Inner, Numeric, Real, Integer, Signed, Unsigned },
  ops::{ NonOps, BaseOps, NumericOps, RealOps, BaseHops, NumericHops, RealHops },
};


/// Multidimensional array.
///
/// Tensors may contain any type that satisfies [Inner], but
/// additional methods are available for [Numeric], [Real], [Integer], [Signed], [Unsigned]
/// and [boolean](bool) inner types.
///
/// [Real] tensor types can be wrapped in a [Variable] by
/// calling [tracked](Tensor::tracked) or [trained](Tensor::trained).

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor<T: Inner> {
  shape: Shape,
  data: RcCell<Vec<T>>,
}

impl<T: Inner> BaseHops<T> for Tensor<T> {}
impl<T: Numeric> NumericHops<T> for Tensor<T> {}
impl<T: Real> RealHops<T> for Tensor<T> {}

impl<T: Inner> PartialEq for Tensor<T> {
  fn eq(&self, rhs: &Self) -> bool {
    if self.shape.dims != rhs.shape.dims { return false }
    for (l, r) in self.param_iter().zip(rhs.param_iter()) {
      if l != r { return false }
    }
    true
  }
}

impl<T: Real> From<T> for Tensor<T> {
  fn from(value: T) -> Self {
    Self::scalar(value)
  }
}

impl<T: Inner> Tensor<T> {
  pub(crate) fn from_shared(shape: Shape, other: &Self) -> Self {
    Self { shape, data: other.data.clone() }
  }

  pub(crate) fn from_shape_raw(shape: Shape, data: Vec<T>) -> Self {
    Self { shape, data: make_rc_cell(data) }
  }

  pub fn from_shape(shape: Shape, data: Vec<T>) -> Self {
    debug_assert!(shape.size_raw() == data.len(),
      "{} doesn't match data length {}", shape, data.len());

    Self::from_shape_raw(shape, data)
  }

  pub fn new(shape: &[usize], data: Vec<T>) -> Self {
    Self::from_shape(Shape::new(shape), data)
  }

  pub fn vec(vec: &[T]) -> Self {
    assert!(vec.len() >= 1, "Cannot create Tensor from empty slice");
    Self::new(&[vec.len()], vec.to_vec())
  }

  pub fn from_vec(vec: Vec<T>) -> Self {
    assert!(vec.len() >= 1, "Cannot create Tensor from empty vector");
    Self::new(&[vec.len()], vec)
  }

  pub fn init(shape: &[usize], mut cb: impl FnMut(Vec<usize>) -> T) -> Self {
    let shape = Shape::new(shape);
    // let data = (0..shape.size()).map(|i| cb(shape.indices(i)) ).collect();
    let data = (0..shape.size()).map(|_| cb(vec![]) ).collect(); //XXX
    Self::from_shape(shape, data)
  }

  pub fn linearize<O: Inner>(tensors: &[Self], cb: impl Fn(Vec<T>) -> O) -> Tensor<O> {
    let first = tensors.first().expect("Cannot linearize empty list");
    let shape = Shape::new(&first.shape.dims);
    let mut iters: Vec<_> = tensors.iter().map(|t| t.param_iter() ).collect();
    let data = (0..shape.size()).map(|_| {
      cb(iters.iter_mut().map(|iter| iter.next().unwrap() ).collect::<Vec<_>>())
    }).collect();
    Tensor::from_shape(shape, data)
  }

  pub fn raw(&self) -> RefT<Vec<T>> {
    borrow(&self.data)
  }

  pub fn raw_mut(&self) -> RefMutT<Vec<T>> {
    borrow_mut(&self.data)
  }

  #[cfg(not(feature = "threading"))]
  pub fn into_raw(self) -> Vec<T> {
    Rc::unwrap_or_clone(self.data).into_inner()
  }

  #[cfg(feature = "threading")]
  pub fn into_raw(self) -> Vec<T> {
    self.data.read().clone()
  }

  pub fn shared_with(&self, other: &Self) -> bool {
    RcT::ptr_eq(&self.data, &other.data)
  }

  pub fn extract(&self) -> Vec<T> {
    self.param_iter().collect()
  }

  pub fn contiguous(&self) -> Self {
    if self.shape.contiguous() {
      self.clone()
    } else {
      self.detach()
    }
  }

  pub fn is_complete(&self) -> bool {
    self.shape.complete() && (self.size() == self.raw().len())
  }

  pub fn complete(&self) -> Self {
    if self.is_complete() {
      self.clone()
    } else {
      self.detach()
    }
  }

  pub fn detach(&self) -> Self {
    self.vectorize(|a| a )
  }

  pub fn zip<O,F>(&self, rhs: &Self, cb: F) -> Tensor<O>
  where
    O: Inner,
    F: Fn((T, T)) -> O,
  {
    let lhs = self.broadcast(&rhs.shape, None);
    let rhs = rhs.broadcast(&self.shape, None);
    let data: Vec<O> = lhs.param_iter()
      .zip(rhs.param_iter())
      .map(cb)
      .collect();
    Tensor::new(&rhs.shape.dims, data)
  }

  // zip_over(dim)

  pub fn vectorize<O,F>(&self, cb: F) -> Tensor<O>
  where
    O: Inner,
    F: FnMut(T) -> O,
  {
    let data = self.param_iter().map(cb).collect();
    Tensor::new(&self.shape.dims, data)
  }

  pub fn reduce<F>(&self, cb: F) -> Option<Self>
  where
    F: Fn(T, T) -> T,
  {
    let item = self.param_iter().reduce(cb);
    item.and_then(|item| Some(Tensor::new(&[], vec![item])) )
  }

  pub fn collapse<O,F>(&self, dim: isize, cb: F) -> Tensor<O>
  where
    O: Inner,
    F: Fn(Self) -> O,
  {
    let dim = negative_index(dim, self.shape.rank(), false);
    let data = self.unsqueeze(0).iter(dim as isize)
      .map(cb)
      .collect();
    Tensor::new(&self.shape.dims[..dim], data)
  }

  pub fn collapse_only<O,F>(&self, dim: isize, cb: F) -> Tensor<O>
  where
    O: Inner,
    F: Fn(Self) -> Tensor<O>,
  {
    let dim = negative_index(dim, self.shape.rank(), false);
    let data = self.unsqueeze(0).iter(dim as isize)
      .flat_map(|t| cb(t).complete().into_raw() )
      .collect();
    let mut dims = self.shape.dims.clone();
    dims[dim] = 1;
    Tensor::new(&dims, data)
  }

  pub fn expand<O,F>(&self, cb: F) -> Tensor<O>
  where
    O: Inner,
    F: Fn(T) -> Vec<O>,
  {
    let mut len = -1;
    let data: Vec<O> = self.param_iter()
      .map(cb)
      .inspect(|vec| {
        debug_assert!(len == -1 || vec.len() == len as usize,
          "Expansion must produce arrays of equal size");
        len = vec.len() as isize;
      })
      .collect::<Vec<_>>()
      .concat();
    let mut dims = self.shape.dims.clone();
    dims.push(len as usize);
    Tensor::new(&dims, data)
  }

  pub fn map<O,F>(&self, dim: isize, cb: F) -> Tensor<O>
  where
    O: Inner,
    F: Fn(Self) -> Tensor<O>,
  {
    let dim = negative_index(dim, self.shape.rank(), false);
    let mut dims = None;
    let data = self.iter(dim as isize)
      .map(cb)
      .inspect(|t| {
        debug_assert!(dims.is_none() || &t.shape.dims == dims.as_ref().unwrap(),
          "Map must produce tensors of equal dimensions");
        if !dims.is_some() {
          dims = Some(t.shape.dims.clone());
        }
      })
      .flat_map(|t| t.complete().into_raw() )
      .collect();
    let dims = vec![self.shape.dims[..=dim].to_vec(), dims.unwrap()].concat();
    Tensor::new(&dims, data)
  }

  pub fn iter(&self, dim: isize) -> TensorSliceIterator<T> {
    TensorSliceIterator::new(self, dim)
  }

  pub fn param_iter(&self) -> TensorIterator<T> {
    TensorIterator::new(self)
  }

  // pub fn at(&self, indices: &[usize]) -> Self {
  //   let shape = self.shape.take(indices);
  //   let data = self.data.clone();
  //   Self { shape, data}
  // }

  pub fn op_assign(&self, other: &Self, cb: impl Fn(&mut T, T)) {
    if self.shared_with(other) { panic!("Tensor was fed from shared storage") }
    //XXX check if RC has other references and copy data if so
    let mut data = self.raw_mut();
    let other_data = other.raw();
    let other_shape = other.shape.broadcast(&self.shape, None);
    debug_assert!(self.shape.squeeze_all().dims == other_shape.squeeze_all().dims,
      "Could not assign {} tensor to {} tensor", other.shape, self.shape);
    for (i, j) in self.shape.iter().zip(other_shape.iter()) {
      cb(&mut data[i], other_data[j].clone());
    }
  }

  pub fn assign(&self, other: &Self) {
    self.op_assign(other, |a, b| *a = b )
  }

  pub fn refill(&self, filler: T) {
    let mut data = self.raw_mut();
    for i in self.shape.iter() {
      data[i] = filler.clone();
    }
  }

  // pub fn set(&mut self, indices: &[isize], other: &Self) {
  //   self.at(indices).assign(other)
  // }

  pub fn view(&self, shape: &[usize]) -> Self {
    let shape = self.shape.view(shape);
    let data = self.data.clone();
    Self { shape, data }
  }

  pub fn extend_front(&self, size: usize) -> Self {
    let shape = self.shape.extend_front(size);
    let data = self.data.clone();
    Self { shape, data }
  }

  pub fn transpose_vec(&self, extend_front: bool) -> Self {
    let mut shape = self.shape.clone();
    if shape.rank() == 1 {
      shape = shape.unsqueeze(if extend_front { -2 } else { -1 })
    }
    let shape = shape.transpose(-1, -2);
    let data = self.data.clone();
    Self { shape, data }
  }

  pub fn equal(&self, rhs: &Self) -> Tensor<bool> {
    //XXX make scalar rhs possible using Into<Tensor>
    self.zip(rhs, |(a, b)| a == b )
  }

  pub fn to_vec(&self, dim: isize) -> Vec<Self> {
    self.iter(dim).collect()
  }

  pub fn shuffle(&self, dim: isize) -> Self {
    self.unsqueeze(0).map(dim, |sub| {
      let mut sub = sub.to_vec(0);
      sub.shuffle(&mut rand::thread_rng());
      Self::rows(&sub)
    }).squeeze_only(0)
  }
}

// Numeric

impl<T: Numeric> Tensor<T> {
  pub fn arrange(shape: &[usize], start: T, step: T) -> Self {
    Self::new(shape, (0..shape.iter().product())
      .map(|i| T::from(i).unwrap() * step + start )
      .collect())
  }

  pub fn hot_encode(idx: usize, size: usize) -> Self {
    let mut a = vec![T::zero(); size];
    a[idx] = T::one();
    Self::from_vec(a)
  }

  pub fn band(dims: &[usize], num_lower: isize, num_upper: isize) -> Self {
    let shape = Shape::new(dims);
    let mut data = vec![T::zero(); shape.size()];
    let l = shape[-1] as isize;
    let p = shape[-2] as isize;
    for i in 0..shape.size() as isize {
      let n = i % l;
      let m = (i / l) % p;
      let one = (num_lower < 0 || (m - n) <= num_lower) &&
                (num_upper < 0 || (n - m) <= num_upper);
      if one { data[i as usize] = T::one() };
    }
    Self::from_shape(shape, data)
  }

  pub(crate) fn add(&self, rhs: &Self) -> Self {
    self.zip(rhs, |(a, b)| a + b )
  }

  pub(crate) fn sub(&self, rhs: &Self) -> Self {
    self.zip(rhs, |(a, b)| a - b )
  }

  pub(crate) fn mul(&self, rhs: &Self) -> Self {
    self.zip(rhs, |(a, b)| a * b )
  }

  pub(crate) fn div(&self, rhs: &Self) -> Self {
    self.zip(rhs, |(a, b)| a / b )
  }

  pub(crate) fn rem(&self, rhs: &Self) -> Self {
    self.zip(rhs, |(a, b)| a % b )
  }

  //XXX Should be differentiable
  pub fn sum_over(&self, dim: isize) -> Self {
    let dim = negative_index(dim, self.rank(), false);
    if dim == self.rank() - 1 {
      // Optimize basic case
      self.sum(-1).unsqueeze(-1)
    } else {
      self.collapse_only(dim as isize, |t| {
        t.iter(0).sum()
      })
    }
  }

  pub fn gt(&self, rhs: &Self) -> Tensor<bool> {
    self.zip(rhs, |(a, b)| a > b )
  }

  pub fn lt(&self, rhs: &Self) -> Tensor<bool> {
    self.zip(rhs, |(a, b)| a < b )
  }

  pub fn top_k(&self, k: usize, dim: isize) -> Self {
    let dim = negative_index(dim, self.shape.rank() - 1, true);
    self.unsqueeze(0).map(dim as isize, |t| {
      let mut data = t.extract();
      data.sort_by(|a, b| b.partial_cmp(a).unwrap() );
      Self::new(&[k], data[..k].to_vec())
    }).squeeze_only(0)
  }

  /// Collapse dimension using index of its greatest value

  pub fn argmax<O: Integer + Unsigned>(&self, dim: isize) -> Tensor<O> {
    self.collapse(dim, |values| {
      O::from(values.param_iter()
        .enumerate()
        .max_by(|(_,a), (_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) )
        .map(|(idx, _)| idx )
        .unwrap()
      ).unwrap()
    })
  }

  pub fn argmin<O: Integer + Unsigned>(&self, dim: isize) -> Tensor<O> {
    self.collapse(dim, |values| {
      O::from(values.param_iter()
        .enumerate()
        .min_by(|(_,a), (_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) )
        .map(|(idx, _)| idx )
        .unwrap()
      ).unwrap()
    })
  }

  pub fn argmax_over<O: Integer + Unsigned>(&self, dim: isize) -> Tensor<O> {
    self.collapse_only(dim as isize, |t| {
      let rows: Vec<_> = t.iter(0).collect();
      Tensor::linearize(&rows, |col| {
        let value: usize = col.iter()
          .enumerate()
          .max_by(|(_,a), (_,b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal) )
          .map(|(idx, _)| idx )
          .unwrap();
        O::from(value).unwrap()
      })
    })
  }

  //XXX Accept tensors as bounds, replace #max_with
  pub fn clamp(&self, min: T, max: T) -> Self {
    self.vectorize(|a| if a < min { min } else if a > max { max } else { a } )
  }

  pub fn cast<I: Numeric>(&self) -> Tensor<I> {
    self.vectorize(|a| I::from(a).unwrap() )
  }

  pub fn bool(&self, threshold: T) -> Tensor<bool> {
    self.vectorize(|a| a > threshold )
  }
}

impl<T: Numeric> std::ops::AddAssign for Tensor<T> {
  fn add_assign(&mut self, rhs: Self) {
    self.op_assign(&rhs, |a, b| *a += b );
  }
}

impl<T: Numeric> std::ops::SubAssign for Tensor<T> {
  fn sub_assign(&mut self, rhs: Self) {
    self.op_assign(&rhs, |a, b| *a -= b );
  }
}

impl<T: Numeric> std::ops::MulAssign for Tensor<T> {
  fn mul_assign(&mut self, rhs: Self) {
    self.op_assign(&rhs, |a, b| *a *= b );
  }
}

impl<T: Numeric> std::ops::DivAssign for Tensor<T> {
  fn div_assign(&mut self, rhs: Self) {
    self.op_assign(&rhs, |a, b| *a /= b );
  }
}

impl<T: Numeric> std::iter::Sum for Tensor<T> {
  fn sum<I: Iterator<Item = Self>>(iter: I) -> Self where I: Iterator {
    iter.fold(Self::zeros(&[1]), |acc, a| a + acc )
  }
}

// Real

impl<T: Real> Tensor<T> {
  pub fn randr(shape: &[usize], min: T, max: T) -> Self {
    let mut rng = rand::thread_rng();
    Self::init(shape, |_| rng.gen_range(min, max) )
  }

  pub fn rand(shape: &[usize]) -> Self {
    Self::randr(shape, T::zero(), T::one())
  }

  pub fn randn(shape: &[usize]) -> Self {
    let len = shape.iter().product();
    let mut data = vec![T::zero(); len];
    for i in 0..(len as f64 / 2.0).ceil() as usize {
      let j = i * 2;
      let (r1, r2): (T, T) = randn();
      data[j] = r1;
      data[(j + 1) % len] = r2;
    }
    Self::new(shape, data)
  }

  pub fn glorot_uniform(shape: &[usize]) -> Self {
    let limit = T::from((6.0 / (shape[0] + shape[1]) as f64).sqrt()).unwrap();
    Tensor::randr(shape, -limit, limit)
  }

  pub fn glorot_normal(shape: &[usize]) -> Self {
    let gain = T::from((2.0 / (shape[0] + shape[1]) as f64).sqrt()).unwrap();
    Tensor::randn(shape) * gain
  }

  pub fn linspace(shape: &[usize], start: T, end: T) -> Self {
    let size = T::from(shape.iter().product::<usize>()).unwrap();
    Self::arrange(shape, start, (end - start) / (size - T::one()))
  }

  pub fn sine_encoding(seq_len: usize, dim_model: usize, max_wavelength: Option<T>) -> Self {
    let half_dim = dim_model / 2;
    let max_wavelength = max_wavelength.unwrap_or(T::from(10_000.0).unwrap());
    let positions = Self::arrange(&[seq_len, 1], T::zero(), T::one());
    let depths = Self::arrange(&[1, half_dim], T::zero(), T::one()) / T::from(half_dim).unwrap();
    let angle_rates = Self::scalar(T::one()) / Self::scalar(max_wavelength).pow(&depths);
    let rads = positions * angle_rates;
    rads.sin().unsqueeze(-1)
      .concat(&rads.cos().unsqueeze(-1), -1)
      .reshape(&[seq_len, dim_model])
  }

  pub fn bernoulli<O: Numeric>(&self) -> Tensor<O> {
    let mut rng = rand::thread_rng();
    self.vectorize(|a| if rng.gen_range(T::zero(), T::one()) < a {
      O::one()
    } else {
      O::zero()
    })
  }

  //XXX Should collapse last dimension and return Tensor
  pub fn sample(&self) -> usize {
    let mut rng = rand::thread_rng();
    let dist = WeightedIndex::new(&self.cast::<f32>().complete().into_raw()).unwrap();
    dist.sample(&mut rng)
  }

  pub fn tracked(&self) -> Variable<T> {
    Variable::from_tensor(self.clone(), false)
  }

  pub fn trained(&self) -> Variable<T> {
    Variable::from_tensor(self.detach(), true)
  }

  pub(crate) fn input(&self, traintape: RcCell<Traintape<T>>) -> Variable<T> {
    Variable::from_tape(false, &traintape, || self.clone())
  }
}

// Integer

impl<T: Integer> Tensor<T> {
  pub fn accuracy<O: Real>(&self, labels: &Self, num_classes: usize) -> O {
    self
    .equal(&labels)
    .numeric::<O>()
    .sum(0)
    .item() / O::from(labels.dim(0) * num_classes).unwrap()
  }
}

// Unsigned Integer

impl<T: Integer + Unsigned> Tensor<T> {
  pub fn one_hot<O: Numeric>(&self, size: usize) -> Tensor<O> {
    self.expand(|a| {
      let mut hot = vec![O::zero(); size];
      let i: usize = NumCast::from(a).unwrap();
      hot[i] = O::one();
      hot
    })
  }

  pub fn multi_hot<O: Numeric>(&self, size: usize) -> Tensor<O> {
    self.one_hot(size).sum_over(-2).squeeze(&[-2])
  }

  pub fn confusion(&self, labels: &Self) -> Self {
    let num_classes = <usize as NumCast>::from(labels.max(0).item()).unwrap() + 1;
    let confusion = Self::zeros(&[num_classes, num_classes]);
    for (pred, real) in self.iter(0).zip(labels.iter(0)) {
      let mut x = confusion.at(&[
        <isize as NumCast>::from(pred.item()).unwrap(),
        <isize as NumCast>::from(real.item()).unwrap()
      ]);
      x += Tensor::scalar(T::one());
    }
    confusion
  }
}

// Signed

impl<T: Signed> Tensor<T> {
  pub fn signum(&self) -> Self {
    self.vectorize(|a| a.signum() )
  }
}

// Bool

impl Tensor<bool> {
  pub fn numeric<O: Numeric>(&self) -> Tensor<O> {
    self.vectorize(|a| if a { O::one() } else { O::zero() })
  }

  pub fn and(&self, rhs: &Self) -> Self {
    self.zip(rhs, |(a, b)| a && b )
  }

  pub fn or(&self, rhs: &Self) -> Self {
    self.zip(rhs, |(a, b)| a || b )
  }

  //XXX choose dimension
  pub fn all(&self) -> Option<bool> {
    self.reduce(|acc, a| acc && a ).and_then(|value| Some(value.item()) )
  }

  pub fn any(&self) -> Option<bool> {
    self.reduce(|acc, a| acc || a ).and_then(|value| Some(value.item()) )
  }

  //XXX Should be differentiable
  pub fn when<O: Inner>(&self, either: Tensor<O>, or: Tensor<O>) -> Tensor<O> {
    let shape = self.shape.broadcast(&either.shape, None).broadcast(&or.shape, None);
    let this = self.broadcast(&shape, None);
    let either = either.broadcast(&shape, None);
    let or = or.broadcast(&shape, None);
    let data = either.param_iter()
      .zip(or.param_iter())
      .zip(this.param_iter())
      .map(|((e, o), b)| if b { e } else { o } )
      .collect();
    Tensor::from_shape(shape, data)
  }
}

impl std::ops::Not for Tensor<bool> {
  type Output = Self;

  fn not(self) -> Self::Output {
    self.vectorize(|a| !a )
  }
}

// Display

impl<T: Inner> std::fmt::Display for Tensor<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
    write!(f, "Tensor{:?} ", self.shape.dims)?;
    print_chunks(0, &self.shape, &self.detach().raw(), f)?;
    Ok(())
  }
}

fn print_chunks<T: std::fmt::Debug>(idx: usize, shape: &Shape, vec: &[T], f: &mut std::fmt::Formatter) -> std::fmt::Result {
  let indent = (0..idx * 2).map(|_| " ").collect::<String>();
  if shape.rank() == 0 {
    write!(f, "{indent}{:?}", vec[0])?;
  } else if idx == shape.rank() - 1 {
    write!(f, "{indent}{:?}\n", vec)?;
  } else {
    let chunks = vec.chunks(vec.len() / shape.dims[idx]);
    write!(f, "{indent}[\n")?;
    for chunk in chunks {
      print_chunks(idx + 1, shape, chunk, f)?;
    }
    write!(f, "{indent}]\n")?;
  }
  Ok(())
}


/// Iterate slices of a [Tensor] along a particular axis.

pub struct TensorSliceIterator<'a, T: Inner> {
  tensor: &'a Tensor<T>,
  iter: DimensionIterator,
}

impl<'a, T: Inner> TensorSliceIterator<'a, T> {
  fn new(tensor: &'a Tensor<T>, dim: isize) -> Self {
    let dim = negative_index(dim, tensor.rank(), false);
    let dims = &tensor.shape.dims[0..dim + 1];
    Self {
      tensor,
      iter: DimensionIterator::new(dims),
    }
  }
}

impl<T: Inner> Iterator for TensorSliceIterator<'_, T> {
  type Item = Tensor<T>;

  fn next(&mut self) -> Option<Self::Item> {
    let dims = self.iter.next();
    dims.and_then(|dims| Some(self.tensor.at(&dims)) )
  }
}


/// Iterate a [Tensor]'s individual parameters.

pub struct TensorIterator<'a, T: Inner> {
  data: RefT<'a, Vec<T>>,
  shape_iter: SwitchIterator<'a>,
}

impl<'a, T: Inner> TensorIterator<'a, T> {
  pub fn new(tensor: &'a Tensor<T>) -> Self {
    Self {
      data: tensor.raw(),
      shape_iter: tensor.shape.iter(),
    }
  }
}

impl<T: Inner> Iterator for TensorIterator<'_, T> {
  type Item = T;

  fn next(&mut self) -> Option<Self::Item> {
    self.shape_iter.next().map(|i| self.data[i].clone())
  }
}


#[cfg(test)]
mod tests {
  use super::*;

  #[test]
  fn index() {
    let x = Tensor::new(&[2,2,2], vec![1, 2, 3, 4, 5, 6, 7, 8]);
    assert_eq!(x.at(&[0,0]), Tensor::vec(&[1, 2]));
    assert_eq!(x.at(&[1,1]), Tensor::vec(&[7, 8]));
    assert_eq!(x.at(&[0,1,1]), Tensor::scalar(4));
    assert_eq!(x.at(&[0]), Tensor::new(&[2,2], vec![1, 2, 3, 4]));
  }

  #[test]
  fn range() {
    let x = Tensor::vec(&[3, 5, 6]);
    assert_eq!(x.range(&[1..-1]), Tensor::vec(&[5, 6]));

    let x = Tensor::new(&[2, 3], vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(x.range(&[0..-1, 1..-1]), Tensor::new(&[2,2], vec![2, 3, 5, 6]));
  }

  #[test]
  fn transpose_vec() {
    let a = Tensor::arrange(&[3], 1, 1).transpose_vec(true);
    assert_eq!(a.shape.dims, vec![3,1]);
  }

  #[test]
  fn broadcast() {
    let x = Tensor::new(&[1,2,3], vec![1, 2, 3, 4, 5, 6]);

    let y = Tensor::new(&[    1], vec![1]);
    assert_eq!(&x + y, Tensor::new(&[1,2,3], vec![2, 3, 4, 5, 6, 7]));

    let y = Tensor::new(&[    3], vec![1, 2, 3]);
    assert_eq!(&x + y, Tensor::new(&[1,2,3], vec![2, 4, 6, 5, 7, 9]));

    let y = Tensor::new(&[  2,3], vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(&x + y, Tensor::new(&[1,2,3], vec![2, 4, 6, 8, 10, 12]));

    let y = Tensor::new(&[  2,1], vec![1, 2]);
    assert_eq!(&x + y, Tensor::new(&[1,2,3], vec![2, 3, 4, 6, 7, 8]));
  }

  #[test]
  fn sum_over() {
    let a = Tensor::arrange(&[3,2,2], 0, 1).sum_over(1);
    assert_eq!(a, Tensor::new(&[3, 1, 2], vec![2, 4, 10, 12, 18, 20]));

    let a = Tensor::arrange(&[3,2,2], 0, 1).sum_over(-1);
    assert_eq!(a, Tensor::new(&[3, 2, 1], vec![1, 5, 9, 13, 17, 21]));
  }

  #[test]
  fn argmax() {
    let a: Tensor<usize> = Tensor::arrange(&[3,2,2], 0, 1).argmax(-1);
    assert_eq!(a, Tensor::new(&[3, 2], vec![1, 1, 1, 1, 1, 1]));
  }

  #[test]
  fn map() {
    let a: Tensor<usize> = Tensor::<usize>::ones(&[3,2]).map(0, |_| Tensor::ones(&[4,4]) );
    assert_eq!(a.shape.dims, vec![3,4,4]);

    let a: Tensor<usize> = Tensor::<usize>::ones(&[3,2]).map(-1, |_| Tensor::ones(&[4,4]) );
    assert_eq!(a.shape.dims, vec![3,2,4,4]);
  }

  #[test]
  fn top_k() {
    let a: Tensor<usize> = Tensor::arrange(&[5,5], 0, 1).top_k(3, 0);
    assert_eq!(a, Tensor::new(&[3], vec![24, 23, 22]));

    let a: Tensor<usize> = Tensor::arrange(&[5,5], 0, 1).top_k(3, -1);
    assert_eq!(a, Tensor::new(&[5,3], vec![4, 3, 2, 9, 8, 7, 14, 13, 12, 19, 18, 17, 24, 23, 22]));

    let a: Tensor<usize> = Tensor::arrange(&[5], 0, 1).top_k(3, -1);
    assert_eq!(a, Tensor::new(&[3], vec![4, 3, 2]));
  }

  #[test]
  fn shuffle() {
    let a = Tensor::arrange(&[3,2,2], 0, 1);
    let b = a.shuffle(1);
    assert_eq!(b.shape.dims, a.shape.dims);
  }
}
