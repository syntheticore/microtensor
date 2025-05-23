// This example demonstrates building a simple multi layer perceptron
// in a functional style using the Layer API.

use microtensor::{ ops::*, Tensor, Variable, Module, Layer };

fn mlp(input: &Variable<f32>) -> Variable<f32> {
  input
    .dense(16)
    .relu()
    .dense(10)
    .sigmoid()
}

fn main() {
  // Use the ::simple contructor for modules with a single input/output
  let mut model = Module::simple(mlp);

  // --- Insert real data here ---
  let input = Tensor::ones(&[32, 28 * 28]);
  let labels = (Tensor::rand(&[32]) * 10.0).cast::<u8>().one_hot(10);

  // Retrieve the first and only output
  let pred = model.run_traced(0, &[&input]); // Or use ::run to run in eager mode

  // Module output is further differentiable
  let _loss = pred.mse(&labels.tracked());
}
