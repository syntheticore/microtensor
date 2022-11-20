# microtensor

Automatic differentiation for tensor operations.

WIP: Don't use in production!

## Features

- **Safe auto-grad** -- Non-differentiable operations return a separate
type that cannot be back-propagated, revealing gaps in your computation graph
at compile time.

- **Broadcasting** — Tensors with differing but compatible shapes get
broadcasted to matching dimensions automatically for most operations.

- **Arbitrary inner types** -- Tensors can store *almost* any data type and
compute gradients for any inner type that satisfies [Real](scalar::Real).

- **Zero-copy views** — Tensors may be sliced, indexed, reshaped, transposed and
broadcasted without actually copying any data in most situations.

- **Graph recycling** -- Computation graphs, created by tracing an eager computation,
can be reevaluated at a later time with new input data. They can also be serialized
and loaded elsewhere, without access to the original code.

## Examples

Evaluating and minimizing a non-linear function:
```rust
use microtensor::{prelude::*, Tensor};

// Create variables from tensors
let w = Tensor::randn(&[2, 16]).trained();
let b = Tensor::zeros(&[16]).trained();

for _ in 0..100 {
  // Do some computation
  let x = Tensor::vec(&[1.0, 2.0]).tracked();
  let loss = ((x.mm(&w) + &b).sigmoid() - 0.5).sqr().mean(0);

  // Compute gradients
  loss.backward();

  // Nudge w and b in order to minimize loss
  for mut param in loss.parameters() {
    param -= param.grad().unwrap() * 0.01
  }

  // Reset gradients
  loss.reset()
}
```

Automatic broadcasting:
```rust
use microtensor::{prelude::*, Tensor};

let a = Tensor::arrange(&[2, 16], 0., 1.);
let b = Tensor::ones(&[2]);
let c = &a - b.unsqueeze(-1) + 1.;

assert_eq!(a, c);

```

Generic return types:
```rust
use microtensor::{prelude::*, Tensor};

let t = Tensor::<f32>::randn(&[16]);
let _a: u8  = t.argmax(0).item();
let _b: u16 = t.argmax(0).item(); // argmax will produce a Tensor<u16> here

```

### More examples
Check the `/examples` folder for more example code.

## License
  MIT
