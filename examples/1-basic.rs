use microtensor::{ ops::*, Tensor };

fn main() {
  // Create trainable variables from tensors
  let weights = Tensor::randn(&[2, 8]).trained();
  let bias = Tensor::zeros(&[8]).trained();

  let learning_rate = 0.001;

  // Basic gradient descent
  for _ in 0..100 {

    // Compute loss
    let x = Tensor::vec(&[1.0, 2.0]).tracked();
    let loss = ((x.mm(&weights) + &bias).sigmoid() - 0.5).sqr().mean(0);

    // Compute gradients
    loss.backward();

    println!("Gradient of loss with respect to weights: {}", weights.grad().unwrap());

    // Minimize loss by updating model parameters
    for mut param in loss.parameters() {
      param -= param.grad().unwrap() * learning_rate
    }

    // Reset gradients
    loss.reset();
  }
}
