use microtensor::{ ops::*, Tensor };

fn main() {
  // Create trainable variables from tensors
  let w = Tensor::randn(&[2, 8]).trained();
  let b = Tensor::zeros(&[8]).trained();

  // Basic gradient descent
  for _ in 0..100 {

    // Compute loss
    let x = Tensor::vec(&[1.0, 2.0]).tracked();
    let loss = ((x.mm(&w) + &b).sigmoid() - 0.5).sqr().mean(0);

    // Compute gradients
    loss.backward();

    println!("Gradient of loss with respect to w: {}", w.grad().unwrap());

    // Nudge w and b in order to minimize loss
    for mut param in loss.parameters() {
      param -= param.grad().unwrap() * 0.01
    }

    // Reset gradients
    loss.reset();
  }
}
