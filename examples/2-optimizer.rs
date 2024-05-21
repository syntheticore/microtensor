use microtensor::{ ops::*, Tensor, optimize::{ Optimizer, Adam } };

fn main() {
  // Create trainable variables from tensors
  let weights = Tensor::randn(&[2, 8]).trained();
  let bias = Tensor::zeros(&[8]).trained();

  // Use a standard optimizer
  let mut optimizer = Optimizer::new(0.001, Adam::default());

  // Basic training loop
  for _ in 0..100 {

    // Track training data for compute operations to be recorded
    let x = Tensor::vec(&[1.0, 2.0]).tracked();

    // Compute loss
    let loss = ((x.mm(&weights) + &bias).silu() - 0.5).sqr().mean(0);

    // Back-prop, optimize and reset gradients
    optimizer.minimize(&loss, loss.parameters(), true);
  }
}
