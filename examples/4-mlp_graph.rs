// This example demonstrates building a simple multi layer perceptron
// in a functional style and reusing the resulting computation graph using
// Variable::retrained & GraphModel::run_traced.

// The graph gets created by running all model operations once.
// It is then repeatedly fed with new inputs and recomputed.

// This approach offers best performance, as memory for intermediate
// operations doesn't need to be re-allocated for repeated executions of the graph
// and additional graph-level optimizations can take place.

// ::run_traced will also retrace your code automatically when input dimensions change,
// for instance when switching between batched training and single input inference.

// Note that control statements used during the graph's contruction will
// be *baked in* and don't get re-evaluated in subsequent runs when using ::run_traced.

// Use ::run instead, to have your code re-executed for every run of the model.

use microtensor::{ ops::*, Tensor, Variable, Module, Layer };

fn dense_layer(input: &Variable<f32>, size: usize) -> Variable<f32> {
  let weights = input.retrained(|| Tensor::randn(&[input.dim(-1), size]) / size as f32 );
  let bias = input.retrained(|| Tensor::zeros(&[size]) ); // The constructor given to ::retrained will only run once & cache the trained weights
  input.mm(&weights) + bias
}

fn perceptron(input: &Variable<f32>) -> Variable<f32> {
  let out = dense_layer(input, 16).relu();
  dense_layer(&out, 10).sigmoid()
}

fn main() {

// This example uses a standard module with multiple inputs/outputs and calculates the loss
// within the module. /examples/5-mlp-layer.rs demonstrates a simpler setup.
  let mut model = Module::new(|inputs| {
    let pred = perceptron(&inputs[0]);
    if let Some(target) = inputs.get(1) {
      let loss = (&pred - target).sqr().mean(0);
      vec![pred, loss]
    } else {
      vec![pred]
    }
  });

  // --- Insert real data here ---
  let input = Tensor::ones(&[32, 28 * 28]);
  let labels = (Tensor::rand(&[32]) * 10.0).cast::<u8>().one_hot(10);

  // Retrieve the first output for inference
  let _pred = model.run_traced(0, &[&input]); // Or use ::run to run in eager mode

  // Or retrieve the second output for optimization
  let _loss = model.run_traced(1, &[&input, &labels]);
}
