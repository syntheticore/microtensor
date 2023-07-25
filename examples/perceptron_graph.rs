// This example demonstrates building a simple multi layer perceptron
// in a functional style and reusing the resulting computation graph.

// The graph gets created by running all model operations once.
// It is then repeatedly fed with new inputs and recomputed.

// This approach offers the best performance as memory for intermediate
// operations doesn't need to be reallocated for repeated executions of the graph.
// It also makes it easy to save & load the model.

// Note that control statements used during the graph's contruction will
// be *baked in* and don't get reevaluated in subsequent runs.
// If you need that kind of flexibility in your model, please check the
// 'perceptron_eager' example!

use microtensor::{ ops::*, Tensor, Variable };

fn dense_layer(input: &Variable<f32>, size: usize) -> Variable<f32> {
  let weights = (Tensor::randn(&[input.dim(-1), size]) / size as f32).trained();
  let bias = Tensor::zeros(&[size]).trained();
  input.mm(&weights) + bias
}

fn perceptron(input: &Variable<f32>) -> Variable<f32> {
  let output = dense_layer(input, 16).relu();
  dense_layer(&output, 10).sigmoid()
}

fn main() {
  // Define model by performing all computations on a placeholder once
  let image_input = Tensor::zeros(&[32, 28 * 28]).tracked();
  let output = perceptron(&image_input);

  // Define the loss to me minimized
  let label_input = Tensor::zeros(&[32, 10]).tracked();
  let loss = (&label_input - &output).sqr().mean(0);

  // Train with some labeled samples
  let learning_rate = 0.01;
  for _ in 0..100 {
    // Insert real training data here
    let images = Tensor::ones(&[32, 28 * 28]).tracked();
    let labels = (Tensor::rand(&[32]) * 10.0).cast::<u8>().one_hot(10);

    // Feed existing computation graph with new inputs
    image_input.assign(&images);
    label_input.assign(&labels);

    // Recompute output and loss
    loss.forward();

    // Compute gradients
    loss.backward();

    // Minimize loss by updating model parameters
    for mut param in loss.parameters() {
      param -= param.grad().unwrap() * learning_rate
    }

    // Reset gradients
    loss.reset();
  }
}
