use std::io;
use std::fs;
use std::rc::Rc;
use std::collections::HashMap;

use serde::{Serialize, Deserialize, de::DeserializeOwned};

use crate::{
  scalar::Real,
  variable::{ Variable, Node, NodeCell, Op }
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

  pub fn run(&self, inputs: &[&Variable<T>]) {
    for (input, data) in self.inputs.iter().zip(inputs) {
      input.feed(data);
    }
    for output in &self.outputs {
      output.forward(); //XXX Don't calculate entire graph multiple times
    }
  }

  fn history(&self) -> Vec<Rc<Node<T>>> {
    let mut history = self.outputs
      .iter()
      .map(|out| out.history() )
      .collect::<Vec<_>>()
      .concat();
    history.sort_by(|a, b| a.id.partial_cmp(&b.id).unwrap() );
    history.dedup();
    history
  }

  pub fn parameters(&self) -> Vec<Variable<T>> {
    self.history()
      .into_iter()
      .filter(|node| node.cell.trainable )
      .map(|node| Variable { node } )
      .collect()
  }

  pub fn load(filename: &str) -> io::Result<Self> {
    let bytes = fs::read(filename)?;
    let tensor_dump: GraphDump<T> = postcard::from_bytes(&bytes).unwrap();
    let mut nodes: HashMap<usize, Rc<Node<T>>> = HashMap::new();
    for dump in tensor_dump.history {
      let node = Node {
        id: dump.id,
        cell: dump.cell.clone(),
        op: dump.op,
        previous: dump.previous.iter().map(|id| nodes[id].clone() ).collect(),
      };
      nodes.insert(node.id, Rc::new(node));
    }
    let map_tensor = |dump: VariableDump| Variable {
      node: nodes[&dump.node].clone(),
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
      let mut cell = node.cell.clone();
      cell.data = cell.data.detach();
      cell.grad = cell.grad.and_then(|grad| Some(grad.detach()) );
      NodeDump {
        id: node.id,
        cell,
        op,
        previous: node.previous.iter().map(|prev| prev.id ).collect(),
      }
    }).collect();

    let map_tensor = |tensor: &Variable<T>| VariableDump {
      node: tensor.node.id,
    };

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
  cell: NodeCell<T>, //XXX don't save gradients
  op: Option<Op<T>>,
  previous: Vec<usize>,
}

#[derive(Serialize, Deserialize)]
struct VariableDump {
  node: usize,
}

#[derive(Serialize, Deserialize)]
struct GraphDump<T: Real + 'static> {
  history: Vec<NodeDump<T>>,
  inputs: Vec<VariableDump>,
  outputs: Vec<VariableDump>,
}
