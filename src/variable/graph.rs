use std::io;
use std::fs;
use std::thread;
use std::collections::HashMap;
use std::sync::atomic::Ordering;

use itertools::Itertools;
use serde::{ Serialize, Deserialize, de::DeserializeOwned };

use crate::{
  internal::*,
  ops::{ BaseOps, BaseHops },
  scalar::Real,
  shape::Shape,
  tensor::Tensor,
  variable::{ Variable, Node, NodeCell, Op, Traintape, LAST_ID },
};


pub trait Tracer<T: Real>: Fn(&[Variable<T>]) -> Vec<Variable<T>> + Send + Sync + 'static {}
impl<T: Real, F> Tracer<T> for F where F: Fn(&[Variable<T>]) -> Vec<Variable<T>> + Send + Sync + 'static {}

impl<T: Real> std::fmt::Debug for dyn Tracer<T> {
  fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
    write!(f, "<Tracer>")
  }
}


#[derive(Debug)]
pub struct Module<T: Real + 'static> {
  pub graph: Graph<T>,
  tracer: RcT<dyn Tracer<T>>,
  start_count: usize,
  traintape: RcCell<Traintape<T>>,
}

impl<T: Real> Clone for Module<T> {
  fn clone(&self) -> Self {
    Self {
      graph: Graph { inputs: vec![], outputs: vec![] },
      tracer: self.tracer.clone(),
      start_count: self.start_count,
      traintape: make_rc_cell((*borrow(&self.traintape)).clone()),
    }
  }
}

impl<T: Real + Serialize + DeserializeOwned + 'static> Module<T> {
  pub fn new(tracer: impl Tracer<T>) -> Self {
    Self {
      graph: Graph { inputs: vec![], outputs: vec![] },
      tracer: RcT::new(tracer),
      start_count: 0,
      traintape: make_rc_cell(Traintape { tape: vec![], counter: 0 }),
    }
  }

  pub fn simple(tracer: impl Fn(&Variable<T>) -> Variable<T> + Send + Sync + 'static) -> Self {
    Self::new(move |inputs| vec![(tracer)(&inputs[0])] )
  }

  /// Sub-modules are usefull for applying layers repeatedly from within another
  /// enclosing Module. They need to be initialized with an arbitrary [Variable]
  /// that was generated from previous Module inputs.

  pub fn continued(tape_holder: &Variable<T>, tracer: impl Tracer<T>) -> Self {
    let tape = tape_holder.node.traintape.as_ref()
      .expect("Cannot continue Module from a Variable that wasn't generated from Module inputs");
    Self {
      graph: Graph { inputs: vec![], outputs: vec![] },
      tracer: RcT::new(tracer),
      start_count: borrow(tape).counter,
      traintape: tape.clone(),
    }
  }

  pub fn run(&self, output: usize, inputs: &[&Tensor<T>]) -> Variable<T> {
    let inputs: Vec<_> = inputs.into_iter().map(|input| input.tracked() ).collect();
    self.run_raw(output, &inputs.iter().collect::<Vec<_>>())
  }

  pub fn run_raw(&self, output: usize, inputs: &[&Variable<T>]) -> Variable<T> {
    let graph = self.trace(inputs);
    graph.outputs[output].clone()
  }

  pub fn run_traced(&mut self, output: usize, inputs: &[&Tensor<T>]) -> &Variable<T> {
    let retrace =
      self.graph.inputs.len() != inputs.len() ||
      self.graph.inputs
        .iter().zip(inputs)
        .any(|(input, data)| input.dim(0) != data.dim(0) )
    ;
    // Retrace when batch dimensions have changed or on first run
    if retrace {
      let inputs: Vec<_> = inputs.into_iter().map(|input| input.tracked() ).collect();
      self.graph = self.trace(&inputs.iter().collect::<Vec<_>>());
    } else {
      // Update existing graph
      self.graph.run(output, &inputs);
    }
    &self.graph.outputs[output]
  }

  pub fn trace(&self, inputs: &[&Variable<T>]) -> Graph<T> {
    borrow_mut(&self.traintape).counter = self.start_count;
    let inputs: Vec<_> = inputs.into_iter().map(|&input| {
      if input.node.traintape.is_none() {
        input.input(self.traintape.clone())
      } else {
        input.clone()
      }
    }).collect();
    let outputs = (self.tracer)(&inputs);
    Graph::new(&inputs, &outputs)
  }

  pub fn load(&mut self, filename: &str) -> io::Result<()> {
    let bytes = fs::read(filename)?;
    let tape: Traintape<T> = postcard::from_bytes(&bytes)
      .expect(&format!("Could not load graph model from {}", filename));
    let highest_id = tape.tape.iter().map(|node| node.id ).max().unwrap();
    LAST_ID.store(LAST_ID.load(Ordering::Relaxed).max(highest_id + 1), Ordering::Relaxed);
    self.traintape = make_rc_cell(tape);
    Ok(())
  }

  pub fn save(&self, filename: &str) -> io::Result<()> {
    let mut traintape = borrow_mut(&self.traintape);
    traintape.tape = traintape.tape.iter().map(|node| {
      let mut clone = (**node).clone();
      clone.traintape = None;
      RcT::new(clone)
    }).collect();
    let dump = postcard::to_allocvec(&*traintape).unwrap();
    fs::write(filename, dump)
  }

  pub fn multi(self, count: usize) -> MultiModule<T> {
    MultiModule {
      count,
      base_module: self,
    }
  }
}


#[derive(Debug)]
pub struct MultiModule<T: Real> {
  base_module: Module<T>,
  count: usize,
}

impl<T: Real + Serialize + DeserializeOwned> MultiModule<T> {
  pub fn run(&self, output: usize, inputs: &[&Tensor<T>]) -> Variable<T> {
    // Only chunk if batch size permits it
    if output != 1 || inputs[0].dim(0) < self.count { return self.base_module.run(output, inputs) }

    // Chunk all inputs
    let inputs: Vec<Vec<_>> = inputs.into_iter().map(|input| input.chunks(self.count, 0) ).collect();

    // Make sure traintape gets filled on first run, before being copied
    if borrow(&self.base_module.traintape).tape.len() == 0 {
      let chunks = inputs.iter().map(|input| &input[0] ).collect::<Vec<_>>();
      return self.base_module.run(output, &chunks);
    }

    // Run multiple threads, each processing a chunk of the entire batch
    let out: Vec<Variable<T>> = thread::scope(|s| {
      (0..self.count)
        .map(|i| {
          let inputs = &inputs;
          s.spawn(move || {
            // Run a clone of the model with its own traintape counter & graph
            let module = self.base_module.clone();
            let result = module.run(output, &inputs.iter().map(|input| &input[i] ).collect::<Vec<_>>());
            // The generated graphs share only their trained variables (which are leaf nodes),
            // so gradient updates will accumulate in those naturally
            result.backward();
            result
          })
        }).collect::<Vec<_>>().into_iter()
        .map(|h| h.join().unwrap() )
        .collect()
    });

    // Return average loss of all runs
    let len = T::from(out.len()).unwrap();
    out.into_iter().sum::<Variable<T>>() / len
  }

  pub fn load(&mut self, filename: &str) -> io::Result<()> {
    self.base_module.load(filename)
  }

  pub fn save(&self, filename: &str) -> io::Result<()> {
    self.base_module.save(filename)
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
    self.outputs
      .iter()
      .flat_map(|out| out.history() )
      .unique_by(|n| n.id )
      .collect()
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
        traintape: None,
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
