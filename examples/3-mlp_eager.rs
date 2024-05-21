// This example demonstrates building a simple multi layer perceptron
// in an object oriented style, using eager execution.

// All trainable tensors are stored explicitly and the model's code gets
// re-executed for every run of the model, creating a new computation graph every time.

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

  // --- Insert real data here ---
  let input = Tensor::ones(&[32, 28 * 28]);

  // Run the model, creating a fresh computation graph in the process
  let _output = model.run(&input.tracked());
}
