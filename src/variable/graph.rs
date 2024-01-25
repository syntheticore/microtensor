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

  pub fn run(&self, inputs: &[&Tensor<T>]) {
    for (input, data) in self.inputs.iter().zip(inputs) {
      input.assign(data);
    }
    for node in self.history() {
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

  pub fn load(filename: &str) -> io::Result<Self> {
    let bytes = fs::read(filename)?;
    let tensor_dump: GraphDump<T> = postcard::from_bytes(&bytes).unwrap();
    let mut nodes: HashMap<usize, RcT<Node<T>>> = HashMap::new();
    for dump in tensor_dump.history {
      let node = Node {
        id: dump.id,
        cell: NodeCell {
          data: dump.data.complete(),
          grad: dump.grad.and_then(|grad| Some(grad.detach()) )
        },
        op: dump.op,
        previous: dump.previous.iter().map(|id| nodes[id].clone() ).collect(),
        trainable: dump.trainable,
      };
      nodes.insert(node.id, RcT::new(node));
    }
    let map_tensor = |dump: usize| Variable {
      node: nodes[&dump].clone(),
    };
    Ok(Graph {
      inputs: tensor_dump.inputs.into_iter().map(map_tensor).collect(),
      outputs: tensor_dump.outputs.into_iter().map(map_tensor).collect(),
    })
  }

  pub fn save(&self, filename: &str) -> io::Result<()> {
    let history = self.history();

    let history_dump = history.iter().map(|node| {
      let op: Vec<u8> = postcard::to_allocvec(&node.op).unwrap();
      let op = postcard::from_bytes(&op).unwrap();
      NodeDump {
        id: node.id,
        data: if node.trainable { node.cell.data.detach() } else { Tensor::scalar(T::zero()).broadcast(node.cell.data.shape(), None) },
        grad: node.cell.grad.as_ref().and_then(|grad| Some(Tensor::scalar(T::zero()).broadcast(grad.shape(), None)) ),
        op,
        previous: node.previous.iter().map(|prev| prev.id ).collect(),
        trainable: node.trainable,
      }
    }).collect();

    let map_tensor = |tensor: &Variable<T>| tensor.node.id;

    let graph_dump = GraphDump {
      history: history_dump,
      inputs: self.inputs.iter().map(map_tensor).collect(),
      outputs: self.outputs.iter().map(map_tensor).collect(),
    };

    let data: Vec<u8> = postcard::to_allocvec(&graph_dump).unwrap();
    fs::write(filename, data)
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
}

#[derive(Serialize, Deserialize)]
struct GraphDump<T: Real + 'static> {
  history: Vec<NodeDump<T>>,
  inputs: Vec<usize>,
  outputs: Vec<usize>,
}
