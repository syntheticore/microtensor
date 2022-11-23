// This example demonstrates building a computation graph with
// multiple inputs and multiple outputs and saving it to a file.

// The graph can then be loaded elsewhere in its entirety,
// without access to the original code.

use microtensor::{ prelude::*, Tensor, Graph };

fn main() {
  let filename = "model.nn";

  // Build and train a model
  build_model(filename);

  // Load it elsewhere
  load_model(filename);
}

fn build_model(filename: &str) {
  // Have some inputs
  let x1 = Tensor::vec(&[1.0, 2.0]).tracked();
  let x2 = Tensor::ones(&[16]).tracked();
  let w = Tensor::randn(&[2, 16]).trained();

  // Do some computations
  let y = x1.mm(&w);
  let z = (&y * &x2).sum(0);

  // Pack the resulting graph into a Graph structure to make its inputs
  // and outputs explicit and arrange them in an order of your liking.
  let graph = Graph::new(&[x1, x2], &[y, z]);

  // Save entire computation graph to disc
  graph.save(filename).unwrap();
}

fn load_model(filename: &str) {
  let graph = Graph::load(filename).unwrap();

  // Feed new data using #run.
  // Updating the entire graph in this way is more efficient
  // than calling #forward on each individual output.
  graph.run(&[
    &Tensor::vec(&[5.0, 6.0]).tracked(),
    &Tensor::randn(&[16]).tracked(),
  ]);

  // Get new output..
  let z = &graph.outputs[1];
  println!("z is now {}", z.item());

  // ..or train the model further
  let z = &graph.outputs[1];
  z.backward();
  for mut param in z.parameters() {
    param -= param.grad().unwrap() * 0.01
  }
}
