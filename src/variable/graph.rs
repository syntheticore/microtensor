use std::io;
use std::fs;
use std::collections::HashMap;

use itertools::Itertools;
use serde::{Serialize, Deserialize, de::DeserializeOwned};

use crate::{
  internal::*,
  ops::BaseOps,
  scalar::Real,
  variable::{ Variable, Node, NodeCell, Op },
  Tensor,
};


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
        data: if node.trainable || node.op.is_none() { node.cell.data.detach() } else { Tensor::scalar(T::zero()).broadcast(node.cell.data.shape(), None) },
        grad: node.cell.grad.as_ref().and_then(|grad| Some(Tensor::scalar(T::zero()).broadcast(grad.shape(), None)) ),
        op: node.op.clone(),
        previous: node.previous.iter().map(|prev| prev.id ).collect(),
        trainable: node.trainable,
        was_shared: node.op.is_some() && node.previous.len() > 0 && node.cell.data.shared_with(&node.previous[0].cell.data),
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
          data: if dump.was_shared { Tensor::from_shared(dump.data.shape().clone(), &previous[0].cell.data) } else { dump.data.complete() },
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
  grad: Option<Tensor<T>>,
  op: Option<Op<T>>,
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
