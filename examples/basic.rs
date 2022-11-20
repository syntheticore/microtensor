use microtensor::{ prelude::*, Tensor };

fn main() {
  // Define some tensors
  let x = Tensor::vec(&[1.0, 2.0]);
  let w = Tensor::randn(&[2, 8]).trained();
  let b = Tensor::zeros(&[8]).trained();

  // Do some computation
  let z = (x.tracked().mm(&w) + b - 0.5).sqr().mean(0);

  // Compute gradients
  z.backward();

  println!("Gradient of z with respect to w: {}", w.grad().unwrap());

  // Nudge w and b in order to minimize z
  for mut param in z.parameters() {
    param -= param.grad().unwrap() * 0.01
  }
}
