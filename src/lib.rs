//! Automatic differentiation for tensor operations.
//! Tiny. Few dependencies. CPU only. Requires Rust nightly.
//!
//! # Features
//!
//! - **Safe auto-grad** — Non-differentiable operations return a separate
//! type that cannot be back-propagated, revealing gaps in your computation graph
//! at compile time.
//!
//! - **Broadcasting** — Tensors with differing but compatible shapes get
//! broadcasted to matching dimensions automatically for most operations.
//!
//! - **Arbitrary inner types** — Tensors can store *almost* any data type and
//! compute gradients for any inner type that satisfies [scalar::Real].
//!
//! - **Zero-copy views** — Tensors may be sliced, indexed, reshaped, transposed and
//! broadcasted without actually copying any data in most situations.
//!
//! - **Graph recycling** — Computation graphs, created by tracing an eager computation,
//! can be re-evaluated at a later time, with new input data. They can also be serialized
//! and loaded elsewhere, without access to the original code.
//!
//! - **Optimization** — Includes a range of standard optimizers, such as ADAM and Nesterov.
//!
//! # Examples
//!
//! Evaluating and minimizing a non-linear function:
//! ```
//! use microtensor::{ ops::*, Tensor, optimize::{ Optimizer, Adam } };
//!
//! fn main() {
//!   // Create trainable variables from tensors
//!   let w = Tensor::randn(&[2, 8]).trained();
//!   let b = Tensor::zeros(&[8]).trained();
//!
//!   // Use a standard optimizer
//!   let mut optimizer = Optimizer::new(0.001, Adam::default());
//!
//!   // Basic training loop
//!   for _ in 0..100 {
//!
//!     // Track training data for compute operations to be recorded
//!     let x = Tensor::vec(&[1.0, 2.0]).tracked();
//!
//!     // Compute loss
//!     let loss = ((x.mm(&w) + &b).silu() - 0.5).sqr().mean(0);
//!
//!     // Back-prop, optimize and reset gradients
//!     optimizer.minimize(&loss, loss.parameters(), true);
//!   }
//! }
//! ```
//!
//! Generic return types:
//! ```rust
//! use microtensor::{ops::*, Tensor};
//!
//! let t = Tensor::<f32>::randn(&[16]);
//! let _a: u8  = t.argmax(0).item();
//! let _b: u16 = t.argmax(0).item(); // argmax will produce a Tensor<u16> here
//! ```
//!
//! ## More examples
//! Check the `/examples` folder for more example code.
//!
//!
//! # Optional features
//!
//! Some features can be toggled in your `Cargo.toml`.
//!
//! - `unsafe` *(default)* — Accelerated matrix math using [matrixmultiply] crate.
//! - `threading` *(default)* — Thread safety & multi-threaded operation over batch dimensions.
// //! Generic inner types:
// //! ```rust
// //! use microtensor::{ops::*, Tensor};
// //!
// //! let mask: Tensor<bool> = Tensor::randn(&[2, 16]).gt(&Tensor::scalar(1.0)).any(-1);
// //!
// //! assert_eq!(mask.shape().size(), 2);
// //!
// //! ```

#![feature(let_chains)]
#![feature(min_specialization)]

mod internal;
mod shape;
mod tensor;
mod variable;

pub mod ops;
pub mod scalar;
pub mod optimize;

pub use shape::Shape;
pub use tensor::Tensor;
pub use variable::{ Variable, Graph, Module, MultiModule, Layer, UnaryOp, BinaryOp, MultiOp };
