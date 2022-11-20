use std::rc::Rc;
use std::cell::{Ref, RefCell};
use std::ops::Range;
use std::fmt::Debug;

use rand::{Rng};
use num_traits::NumCast;
use serde::{Serialize, Deserialize};

mod cops;
mod lops;

use crate::{
  internal::*,
  shape::Shape,
  variable::Variable,
  scalar::{ Inner, Numeric, Real, Integer, Signed, Unsigned },
  ops::{ BaseOps, NumericOps, Hops },
};


/// Multidimensional array.
///
/// Tensors may contain any type that satisfies [Inner], but
/// additional methods are available for [Numeric], [Real]
/// and [boolean](bool) inner types.
///
/// [Real] tensor types can be wrapped in a [Variable] by
/// calling [tracked](Tensor::tracked) or [trained](Tensor::trained).

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Tensor<T: Inner> {
  shape: Shape,
  data: Rc<RefCell<Vec<T>>>,
}

impl<T: Real> Hops<T> for Tensor<T> {}

impl<T: Inner> PartialEq for Tensor<T> {
  fn eq(&self, rhs: &Self) -> bool {
    if self.shape.squeeze().dims != rhs.shape.squeeze().dims { return false }
    let data_l = self.data.borrow();
    let data_r = rhs.data.borrow();
    for (i, j) in self.shape.iter().zip(rhs.shape.iter()) {
      if data_l[i] != data_r[j] { return false }
    }
    true
  }
}

impl<T: Inner> Tensor<T> {
  pub fn from_shape(shape: Shape, data: Vec<T>) -> Self {
    assert_eq!(shape.size(), data.len(),
      "{} doesn't match data length {}", shape, data.len());
    Self { shape, data: Rc::new(RefCell::new(data)) }
  }

  pub fn new(shape: &[usize], data: Vec<T>) -> Self {
    Self::from_shape(Shape::new(shape), data)
  }

  pub fn vec(vec: &[T]) -> Self {
    Self::new(&[vec.len()], vec.to_vec())
  }

  pub fn from_vec(vec: Vec<T>) -> Self {
    Self::new(&[vec.len()], vec)
  }

  pub fn fill(shape: &[usize], filler: T) -> Self { //XXX store one value only
    Self::new(shape, vec![filler; shape.iter().product()])
  }

  pub fn init(shape: &[usize], mut cb: impl FnMut(Vec<usize>) -> T) -> Self {
    let shape = Shape::new(shape);
    // let data = (0..shape.size()).map(|i| cb(shape.indices(i)) ).collect();
    let data = (0..shape.size()).map(|_| cb(vec![]) ).collect(); //XXX
    Self::from_shape(shape, data)
  }

  pub fn rows(rows: &[Tensor<T>]) -> Self {
    let mut dims = rows[0].shape.dims.clone();
    dims.insert(0, rows.len());
    let data = rows.iter()
      .map(|row| row.detach().into_raw() )
      .collect::<Vec<_>>()
      .concat();
    Self::new(&dims, data)
  }

  pub fn stack(rows: &[Tensor<T>]) -> Self {
    let mut dims = rows[0].shape.dims.clone();
    dims[0] = rows.iter().map(|row| row.shape[0] ).sum();
    let data = rows.iter()
      .map(|row| row.detach().into_raw() )
      .collect::<Vec<_>>()
      .concat();
    Self::new(&dims, data)
  }

  pub fn raw(&self) -> Ref<Vec<T>> {
    self.data.borrow()
  }

  pub fn into_raw(self) -> Vec<T> {
    Rc::unwrap_or_clone(self.data).into_inner()
  }

  pub fn size(&self) -> usize {
    self.shape.size()
  }

  pub fn rank(&self) -> usize {
    self.shape.rank()
  }

  pub fn feed(&self, other: &Self) {
    assert!(self.shape.squeeze().dims == other.shape.squeeze().dims,
      "Could not feed {} tensor with {} tensor", self.shape, other.shape);
    // Avoid clashing borrow when tensors share storage
    let other = if Rc::ptr_eq(&self.data, &other.data) {
      other.detach()
    } else {
      other.clone()
    };
    let mut data = self.data.borrow_mut();
    let other_data = other.data.borrow();
    for (i, j) in self.shape.iter().zip(other.shape.iter()) {
      data[i] = other_data[j];
    }
  }

  pub fn contiguous(&self) -> Self {
    if self.shape.contiguous() {
      self.clone()
    } else {
      self.vectorize(|a| a )
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
    let rhs = rhs.broadcast(self);
    let data: Vec<O> = self.broadcast(&rhs).param_iter()
      .zip(rhs.param_iter())
      .map(cb)
      .collect();
    Tensor::new(&rhs.shape.dims, data)
  }

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

  pub fn collapse_only<F>(&self, dim: isize, cb: F) -> Self
  where
    F: Fn(Self) -> Self,
  {
    let dim = negative_index(dim, self.shape.rank(), false);
    let data = self.unsqueeze(0).iter(dim as isize)
      .flat_map(|t| cb(t).into_raw() )
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
        assert!(len == -1 || vec.len() == len as usize,
          "Expansion must produce arrays of equal size");
        len = vec.len() as isize;
      })
      .collect::<Vec<_>>()
      .concat();
    let mut dims = self.shape.dims.clone();
    dims.push(len as usize);
    Tensor::new(&dims, data)
  }

  pub fn iter(&self, dim: isize) -> TensorSliceIterator<T> {
    TensorSliceIterator::new(self, dim)
  }

  pub fn param_iter(&self) -> TensorIterator<T> {
    TensorIterator::new(self)
  }

  pub fn at(&self, indices: &[usize]) -> Self {
    let shape = self.shape.take(indices);
    let data = self.data.clone();
    Self { shape, data}
  }

  pub fn range(&self, ranges: &[Range<isize>]) -> Self {
    let shape = self.shape.range(ranges);
    let data = self.data.clone();
    Self { shape, data }
  }

  pub fn item(&self) -> T {
    assert!(self.shape.squeeze().rank() == 0,
      "Can't extract item from non-scalar {}", self.shape);
    self.raw()[self.shape.offset]
  }

  pub fn view(&self, shape: &[usize]) -> Self {
    let shape = self.shape.view(shape);
    let data = self.data.clone();
    Self { shape, data }
  }

  pub fn squeeze(&self) -> Self {
    let shape = self.shape.squeeze();
    let data = self.data.clone();
    Self { shape, data }
  }

  pub fn squeeze_only(&self, dim: isize) -> Self {
    let shape = self.shape.squeeze_only(dim);
    let data = self.data.clone();
    Self { shape, data }
  }

  pub fn squeeze_but(&self, dim: isize) -> Self {
    let shape = self.shape.squeeze_but(dim);
    let data = self.data.clone();
    Self { shape, data }
  }

  pub fn transpose(&self, dim1: isize, dim2: isize) -> Self {
    let shape = self.shape.transpose(dim1, dim2);
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
    self.zip(rhs, |(a, b)| a == b )
  }

  pub fn chunks(&self, size: usize) -> Vec<Tensor<T>> {
    let n = self.shape[0] / size;
    let remainder = self.shape[0] % size;
    let slices = (0..n as isize)
      .map(|i| {
        let j = i * size as isize;
        self.range(&[j .. j + size as isize])
      })
      .collect();
    if remainder == 0 {
      slices
    } else {
      let j = n as isize * size as isize;
      [slices, vec![self.range(&[j .. j + remainder as isize])]].concat()
    }
  }
}

impl<T: Numeric> std::iter::Sum for Tensor<T> {
  fn sum<I: Iterator<Item = Self>>(iter: I) -> Self where I: Iterator {
    iter.fold(Self::zeros(&[1]), |acc, a| acc.add(&a) )
  }
}

impl<T: Numeric> Tensor<T> {
  pub fn ones(shape: &[usize]) -> Self {
    Self::new(shape, vec![T::one(); shape.iter().product()])
  }

  pub fn zeros(shape: &[usize]) -> Self { //XXX should only store one zero + strides
    Self::new(shape, vec![T::zero(); shape.iter().product()])
  }

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

  pub fn add(&self, rhs: &Self) -> Self {
    self.zip(rhs, |(a, b)| a + b )
  }

  pub fn sub(&self, rhs: &Self) -> Self {
    self.zip(rhs, |(a, b)| a - b )
  }

  pub fn mul(&self, rhs: &Self) -> Self {
    self.zip(rhs, |(a, b)| a * b )
  }

  pub fn div(&self, rhs: &Self) -> Self {
    self.zip(rhs, |(a, b)| a / b )
  }

  pub fn sum_over(&self, dim: isize) -> Self {
    self.collapse_only(dim, |t| {
      t.iter(0).sum()
    })
  }

  pub fn gt(&self, rhs: &Self) -> Tensor<bool> {
    self.zip(rhs, |(a, b)| a > b )
  }

  pub fn lt(&self, rhs: &Self) -> Tensor<bool> {
    self.zip(rhs, |(a, b)| a < b )
  }

  /// Collapse dimension using index of its greatest value

  pub fn argmax<O: Integer + Unsigned>(&self, dim: isize) -> Tensor<O> {
    self.collapse(dim, |values| {
      let mut max = T::zero();
      let mut index = 0;
      for (i, float) in values.param_iter().enumerate() {
        if float > max {
          max = float;
          index = i;
        }
      }
      O::from(index).unwrap()
    })
  }

  pub fn clamp(&self, min: T, max: T) -> Self {
    self.vectorize(|a| if a < min { min } else if a > max { max } else { a } )
  }

  pub fn cast<I: Numeric>(&self) -> Tensor<I> {
    self.vectorize(|a| I::from(a).unwrap() )
  }
}

impl<T: Numeric> std::ops::AddAssign for Tensor<T> {
  fn add_assign(&mut self, rhs: Self) {
    self.feed(&self.add(&rhs));
  }
}

impl<T: Numeric> std::ops::SubAssign for Tensor<T> {
  fn sub_assign(&mut self, rhs: Self) {
    self.feed(&self.sub(&rhs));
  }
}

impl<T: Numeric> std::ops::MulAssign for Tensor<T> {
  fn mul_assign(&mut self, rhs: Self) {
    self.feed(&self.mul(&rhs));
  }
}

impl<T: Numeric> std::ops::DivAssign for Tensor<T> {
  fn div_assign(&mut self, rhs: Self) {
    self.feed(&self.div(&rhs));
  }
}

impl<T: Real> Tensor<T> {
  pub fn rand(shape: &[usize]) -> Self {
    let mut rng = rand::thread_rng();
    Self::init(shape, |_| rng.gen_range(T::zero(), T::one()) )
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

  pub fn linspace(shape: &[usize], start: T, end: T) -> Self {
    let size = T::from(shape.iter().product::<usize>()).unwrap();
    Self::arrange(shape, start, (end - start) / (size - T::one()))
  }

  pub fn bernoulli<O: Numeric>(&self) -> Tensor<O> {
    let mut rng = rand::thread_rng();
    self.vectorize(|a| if rng.gen_range(T::zero(), T::one()) < a {
      O::one()
    } else {
      O::zero()
    })
  }

  pub fn trained(&self) -> Variable<T> {
    Variable::from_tensor(self.clone(), true)
  }

  pub fn tracked(&self) -> Variable<T> {
    Variable::from(self)
  }

  pub fn log(&self) -> Self {
    self.vectorize(|a| a.ln() )
  }
}

impl<T: Integer> Tensor<T> {
  pub fn accuracy<O: Real>(&self, labels: &Self) -> O {
    self
    .equal(&labels)
    .numeric::<O>()
    .sum(0)
    .item() / O::from(labels.shape()[0]).unwrap()
  }
}

impl<T: Integer + Unsigned> Tensor<T> {
  pub fn one_hot<O: Numeric>(&self, size: usize) -> Tensor<O> {
    self.expand(|a| {
      let mut hot = vec![O::zero(); size];
      let i: usize = NumCast::from(a).unwrap();
      hot[i] = O::one();
      hot
    })
  }

  pub fn confusion(&self, labels: &Self) -> Self {
    let num_classes = <usize as NumCast>::from(labels.max(0).item()).unwrap() + 1;
    let confusion = Self::zeros(&[num_classes, num_classes]);
    for (pred, real) in self.iter(0).zip(labels.iter(0)) {
      let mut x = confusion.at(&[
        <usize as NumCast>::from(pred.item()).unwrap(),
        <usize as NumCast>::from(real.item()).unwrap()
      ]);
      x += Tensor::scalar(T::one());
    }
    confusion
  }
}

impl<T: Signed> Tensor<T> {
  pub fn signum(&self) -> Self {
    self.vectorize(|a| a.signum() )
  }
}

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

  pub fn all(&self) -> Option<bool> {
    self.reduce(|acc, a| acc && a ).and_then(|value| Some(value.item()) )
  }

  pub fn any(&self) -> Option<bool> {
    self.reduce(|acc, a| acc || a ).and_then(|value| Some(value.item()) )
  }
}

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


pub struct TensorSliceIterator<T: Inner> {
  tensor: Tensor<T>,
  index: usize,
}

impl<T: Inner> TensorSliceIterator<T> {
  fn new(tensor: &Tensor<T>, dim: isize) -> Self {
    let dim = negative_index(dim, tensor.shape.rank(), false);
    // Reshape to [1,1,1,0,8]
    let mut shape = tensor.shape.dims.clone();
    for i in 0..dim { shape[i] = 1 }
    shape[dim] = 0;
    let mut tensor = tensor.reshape(&shape);
    for _ in 0..dim { //XXX use squeeze_but
      tensor = tensor.squeeze_only(0);
    }
    Self {
      tensor,
      index: 0,
    }
  }
}

impl<T: Inner> Iterator for TensorSliceIterator<T> {
  type Item = Tensor<T>;

  fn next(&mut self) -> Option<Self::Item> {
    if self.index == self.tensor.shape[0] { return None }
    let out = self.tensor.at(&[self.index]);
    self.index += 1;
    Some(out)
  }
}


pub struct TensorIterator<'a, T: Inner> {
  data: Ref<'a, Vec<T>>,
  shape_iter: Box<dyn Iterator<Item=usize> + 'a>,
}

impl<'a, T: Inner> TensorIterator<'a, T> {
  fn new(tensor: &'a Tensor<T>) -> Self {
    Self {
      data: tensor.data.borrow(),
      shape_iter: tensor.shape.iter(),
    }
  }
}

impl<T: Inner> Iterator for TensorIterator<'_, T> {
  type Item = T;

  fn next(&mut self) -> Option<Self::Item> {
    self.shape_iter.next().and_then(|i| Some(self.data[i]) )
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
    assert_eq!(x.at(&[0,1,1]), Tensor::vec(&[4]));
    assert_eq!(x.at(&[0]), Tensor::new(&[2,2], vec![1, 2, 3, 4]));
  }

  #[test]
  fn range() {
    let x = Tensor::vec(&[3, 5, 6]);
    assert_eq!(x.range(&[1..-1]), Tensor::vec(&[5, 6]));
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
    assert_eq!(x.add(&y), Tensor::new(&[1,2,3], vec![2, 3, 4, 5, 6, 7]));

    let y = Tensor::new(&[    3], vec![1, 2, 3]);
    assert_eq!(x.add(&y), Tensor::new(&[1,2,3], vec![2, 4, 6, 5, 7, 9]));

    let y = Tensor::new(&[  2,3], vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(x.add(&y), Tensor::new(&[1,2,3], vec![2, 4, 6, 8, 10, 12]));
  }

  #[test]
  #[ignore]
  fn pytorch_broadcast() {
    let x = Tensor::new(&[1,2,3], vec![1, 2, 3, 4, 5, 6]);

    let y = Tensor::new(&[  2,1], vec![1, 2, 3, 4, 5, 6]);
    assert_eq!(x.add(&y), Tensor::new(&[1,2,3], vec![2, 3, 4, 5, 6, 7]));
  }

  #[test]
  fn sum_over() {
    let a = Tensor::arrange(&[3,2,2], 0, 1).sum_over(1);
    assert_eq!(a, Tensor::new(&[3, 1, 2], vec![2, 4, 10, 12, 18, 20]));
  }
}
