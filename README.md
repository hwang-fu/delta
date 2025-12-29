# Delta

> A tensor-based automatic differentiation engine built from scratch in Rust.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/Rust-2024%20Edition-orange.svg)](https://www.rust-lang.org/)

---

## Overview

**Delta** is an educational tensor autograd engine that prioritizes clarity and correctness over performance. Built entirely from scratch with zero dependencies, it demonstrates the fundamental concepts behind modern deep learning frameworks.

### Why "Delta"?

- **Nabla (∇)** — the gradient operator
- **Delta (Δ)** — change, difference

Together they represent the core of differentiation: measuring how things change.

---

## Features

### Implemented

- **Tensor Operations**
  - N-dimensional tensor creation and indexing
  - Element-wise arithmetic: `add`, `sub`, `mul`, `div`, `neg`
  - Scalar operations: `scalar_add`, `scalar_mul`
  - Matrix multiplication: `matmul`
  - Transpose: `transpose`, `t()`

- **Operator Overloading**
  - Full support for `+`, `-`, `*`, `/` operators
  - Works with both owned values and references
  - Scalar multiplication: `tensor * 3.0` or `3.0 * tensor`

- **Developer Experience**
  - Pretty-printed tensor display with truncation for large tensors
  - Comprehensive error messages
  - Full test coverage

### More Features to Go

- [ ] Broadcasting for element-wise operations
- [ ] Reduction operations (sum, mean, max)
- [ ] Computation graph with index-based nodes
- [ ] Automatic differentiation (backward pass)
- [ ] Neural network primitives (layers, loss functions, optimizers)
- [ ] C FFI for cross-language support

---

## Quick Start

```rust
use delta::tensor::Tensor;

fn main() {
    // Create tensors
    let a = Tensor::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0], &[2, 3]);
    let b = Tensor::from_vec(vec![7.0, 8.0, 9.0, 10.0, 11.0, 12.0], &[3, 2]);

    // Matrix multiplication
    let c = a.matmul(&b);
    println!("{}", c);

    // Operator overloading
    let x = Tensor::from_vec(vec![1.0, 2.0, 3.0], &[3]);
    let y = Tensor::from_vec(vec![4.0, 5.0, 6.0], &[3]);
    let z = &x + &y;        // Element-wise addition
    let scaled = &x * 2.0;  // Scalar multiplication

    // Transpose
    let t = a.transpose();
    println!("Transposed: {}", t);
}
```

---

## Building

```bash
# Build the library
cargo build

# Run tests
cargo test

# Run the example
cargo run --example basic

# Build documentation
cargo doc --open
```

---

## Project Structure

```
delta/
├── src/
│   ├── lib.rs              # Library root
│   └── tensor/
│       ├── mod.rs          # Module exports
│       ├── shape.rs        # Shape and stride handling
│       ├── storage.rs      # Underlying data storage
│       └── tensor.rs       # Tensor struct and operations
├── examples/
│   └── basic.rs            # Usage examples
└── Cargo.toml
```

---

## Design Philosophy

1. **Educational First** — Code reflects the underlying mathematics
2. **Zero Dependencies** — Built from scratch for learning
3. **Correctness Over Speed** — Naive algorithms, clear implementations
4. **Rust-Idiomatic** — Embraces ownership, uses index-based graphs

---

## Inspiration

- [micrograd](https://github.com/karpathy/micrograd) — Andrej Karpathy's minimal autograd
- [tinygrad](https://github.com/tinygrad/tinygrad) — Simple deep learning framework
- [candle](https://github.com/huggingface/candle) — Hugging Face's Rust ML framework

---

## License

MIT License — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <i>"What I cannot create, I do not understand."</i> — Richard Feynman
</p>
