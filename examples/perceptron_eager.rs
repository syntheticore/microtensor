// This example demonstrates building a simple multi layer perceptron
// in an object oriented style, using eager execution.

// All trainable tensors are stored explicitly and the model's code gets
// reexecuted for every run of the model, creating a new computation graph every time.

// This approach is most flexible and allows for arbitrary control statements
// to be used. If your model doesn't need that kind of flexibility, please
// consider reusing the graph as described in the 'perceptron_graph' example,
// to improve performance.

// Note that saving the computation graph and loading it elsewhere for inference
// might not yield the same results as running the model directly, when using
// this approach.

use microtensor::{ ops::*, Tensor, Variable };

struct DenseLayer {
  weights: Variable<f32>,
  bias: Variable<f32>,
}

impl DenseLayer {
  pub fn new(input_size: usize, size: usize) -> Self {
    Self {
      weights: (Tensor::randn(&[input_size, size]) / size as f32).trained(),
      bias: Tensor::zeros(&[size]).trained(),
    }
  }

  pub fn run(&self, input: &Variable<f32>) -> Variable<f32> {
    input.mm(&self.weights) + &self.bias
  }
}

struct Perceptron {
  hidden: DenseLayer,
  output: DenseLayer,
}

impl Perceptron {
  pub fn new(input_size: usize) -> Self {
    Self {
      hidden: DenseLayer::new(input_size, 16),
      output: DenseLayer::new(16, 10),
    }
  }

  pub fn run(&self, input: &Variable<f32>) -> Variable<f32> {
    let t = self.hidden.run(input).relu();
    self.output.run(&t).sigmoid()
  }
}

fn main() {
  // Construct model that stores all trainable tensors explicitly
  let model = Perceptron::new(28 * 28);

  // Train with labeled samples
  let learning_rate = 0.01;
  for _ in 0..100 {
    // Insert real training data here
    let images = Tensor::ones(&[32, 28 * 28]);
    let labels = (Tensor::rand(&[32]) * 10.0).cast::<u8>().one_hot(10);

    // Run the model, creating a fresh computation graph in the process
    let output = model.run(&images.tracked());

    // Compute loss
    let loss = (&labels.tracked() - &output).sqr().mean(0);

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
