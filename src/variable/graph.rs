use std::io;
use std::fs;
use std::collections::HashMap;

use itertools::Itertools;
use serde::{ Serialize, Deserialize, de::DeserializeOwned };

use crate::{
  internal::*,
  ops::{ BaseOps, BaseHops },
  scalar::Real,
  shape::Shape,
  tensor::Tensor,
  variable::{ Variable, Node, NodeCell, Op },
};


pub struct GraphModel<T: Real + 'static> {
  pub graph: Option<Graph<T>>,
  model: Box<dyn Fn(&[Variable<T>]) -> Vec<Variable<T>>>,
}

impl<T: Real + Serialize + DeserializeOwned + 'static> GraphModel<T> {
  pub fn new(model: impl Fn(&[Variable<T>]) -> Vec<Variable<T>> + 'static) -> Self {
    Self {
      graph: None,
      model: Box::new(model),
    }
  }

  pub fn build_graph(&self, inputs: &[&Tensor<T>]) -> Graph<T> {
    let inputs: Vec<_> = inputs.iter().map(|input| input.tracked() ).collect();
    let outputs = (self.model)(&inputs);
    Graph::new(&inputs, &outputs)
  }

  pub fn run(&mut self, output: usize, inputs: &[&Tensor<T>]) -> &Variable<T> {
    if self.graph.is_none() {
      // First run init
      self.graph = Some(self.build_graph(inputs));
    } else {
      let graph = self.graph.as_ref().unwrap();
      let mut inputs = inputs.to_vec();
      // Insert missing inputs to keep number of graph nodes fixed
      let diff = graph.inputs.len() - inputs.len();
      let start = inputs.len();
      let batch_dim = inputs[0].dim(0); //XXX determine dynamically by comparing dims of primary inputs
      let missing: Vec<_> = (0..diff).map(|i| {
        let j = start + i;
        let mut shape = graph.inputs[j].shape().dims.clone();
        shape[0] = batch_dim;
        Tensor::scalar(T::from(0.0).unwrap()).broadcast(&Shape::new(&shape), None)
      }).collect();
      for dummy in &missing { inputs.push(dummy) }
      if graph.inputs.iter().zip(&inputs).any(|(input, data)| input.dim(0) != data.dim(0) ) {
        // Batch dimension changed -> Re-run full model & replace graph
        let new = self.build_graph(&inputs);
        let new = Self::replace_trained(&new, &graph);
        for node in new.history() { node.forward() }
        self.graph = Some(new);
      } else {
        // Update original graph
        self.graph.as_mut().unwrap().run(output, &inputs);
      }
    }
    &self.graph.as_ref().unwrap().outputs[output]
  }

  fn replace_trained(graph: &Graph<T>, old_graph: &Graph<T>) -> Graph<T> {
    let mut table: HashMap<usize, RcT<Node<T>>> = HashMap::new();
    for (node, old) in graph.history().iter().zip(&old_graph.history()) {
      let mut clone = (** if node.trainable { old } else { node }).clone();
      clone.id = old.id; // Ensure that optimizers can still look up gradients & keep order stable for Graph#history to work
      clone.previous = clone.previous.iter().map(|prev| {
        table.get(&prev.id).unwrap().clone()
      }).collect();
      table.insert(node.id, RcT::new(clone));
    }
    let inputs: Vec<_> = graph.inputs.iter().map(|input| Variable { node: table.get(&input.id()).unwrap().clone() } ).collect();
    let outputs: Vec<_> = graph.outputs.iter().map(|output| Variable { node: table.get(&output.id()).unwrap().clone() } ).collect();
    Graph::new(&inputs, &outputs)
  }
}


/// Snapshot of a computation graph with defined inputs and outputs.
///
/// Can be used for recomputing the entire graph or saving it to disc.

#[derive(Debug, Clone)]
pub struct Graph<T: Real + 'static> {
  pub inputs: Vec<Variable<T>>,
  pub outputs: Vec<Variable<T>>,
}

impl<T: Real + Serialize + DeserializeOwned + 'static> Graph<T> {
  pub fn new(inputs: &[Variable<T>], outputs: &[Variable<T>]) -> Self {
    Self {
      inputs: inputs.into(),
      outputs: outputs.into(),
    }
  }

  pub fn run(&mut self, output: usize, inputs: &[&Tensor<T>]) -> &Variable<T> {
    self.run_history(inputs, self.outputs[output].history());
    &self.outputs[output]
  }

  pub fn run_all(&mut self, inputs: &[&Tensor<T>]) {
    self.run_history(inputs, self.history());
  }

  fn run_history(&self, inputs: &[&Tensor<T>], history: Vec<RcT<Node<T>>>) {
    for (input, data) in self.inputs.iter().zip(inputs) {
      input.assign(data);
    }
    for node in history {
      node.forward();
    }
  }

  fn history(&self) -> Vec<RcT<Node<T>>> {
    let mut history = self.outputs
      .iter()
      .map(|out| out.history() )
      .collect::<Vec<_>>()
      .concat();
    history.sort_by(|a, b| a.id.partial_cmp(&b.id).unwrap() );
    history.into_iter().unique_by(|a| a.id ).collect()
  }

  pub fn serialize(&self) -> Vec<u8> {
    let history = self.history();

    let history_dump = history.iter().map(|node| {
      NodeDump {
        id: node.id,
        data: if node.op.is_none() { node.cell.data.detach() } else { Tensor::scalar(T::zero()).broadcast(node.cell.data.shape(), None) },
        shape: node.cell.data.shape().clone(), // Save original shape for shared tensors
        grad: node.cell.grad.as_ref().and_then(|grad| Some(Tensor::scalar(T::zero()).broadcast(grad.shape(), None)) ),
        op: node.op.clone(),
        previous: node.previous.iter().map(|prev| prev.id ).collect(),
        trainable: node.trainable,
        was_shared: node.op.is_some() && node.cell.data.shared_with(&node.previous[0].cell.data),
      }
    }).collect();

    let map_tensor = |tensor: &Variable<T>| tensor.node.id;

    let graph_dump = GraphDump {
      history: history_dump,
      inputs: self.inputs.iter().map(map_tensor).collect(),
      outputs: self.outputs.iter().map(map_tensor).collect(),
    };

    postcard::to_allocvec(&graph_dump).unwrap()
  }

  pub fn deserialize(bytes: Vec<u8>) -> Self {
    let tensor_dump: GraphDump<T> = postcard::from_bytes(&bytes).unwrap();
    let mut nodes: HashMap<usize, RcT<Node<T>>> = HashMap::new();
    for dump in tensor_dump.history {
      let previous: Vec<_> = dump.previous.iter().map(|id| nodes[id].clone() ).collect();
      let node = Node {
        id: dump.id,
        cell: NodeCell {
          data: if dump.was_shared { Tensor::from_shared(dump.shape, &previous[0].cell.data) } else { dump.data.complete() },
          grad: dump.grad.and_then(|grad| Some(grad.detach()) ),
        },
        op: dump.op,
        previous,
        trainable: dump.trainable,
      };
      nodes.insert(node.id, RcT::new(node));
    }
    let map_tensor = |dump: usize| Variable {
      node: nodes[&dump].clone(),
    };
    Self {
      inputs: tensor_dump.inputs.into_iter().map(map_tensor).collect(),
      outputs: tensor_dump.outputs.into_iter().map(map_tensor).collect(),
    }
  }

  pub fn save(&self, filename: &str) -> io::Result<()> {
    fs::write(filename, self.serialize())
  }

  pub fn load(filename: &str) -> io::Result<Self> {
    Ok(Self::deserialize(fs::read(filename)?))
  }
}

#[derive(Serialize, Deserialize)]
struct NodeDump<T: Real + 'static> {
  id: usize,
  data: Tensor<T>,
  shape: Shape,
  grad: Option<Tensor<T>>,
  op: Option<Op>,
  previous: Vec<usize>,
  trainable: bool,
  was_shared: bool,
}

#[derive(Serialize, Deserialize)]
struct GraphDump<T: Real + 'static> {
  history: Vec<NodeDump<T>>,
  inputs: Vec<usize>,
  outputs: Vec<usize>,
}
