# ndtensor

[![crates.io](https://img.shields.io/crates/v/ndtensor?style=for-the-badge&logo=rust)](https://crates.io/crates/ndtensor)
[![docs.rs](https://img.shields.io/docsrs/ndtensor?style=for-the-badge&logo=docs.rs)](https://docs.rs/ndtensor)
[![GitHub License](https://img.shields.io/github/license/FL03/ndtensor?style=for-the-badge&logo=github)](https://github.com/FL03/ndtensor/blob/main/LICENSE)

***

_**Warning: The library still in development and is not yet ready for production use.**_

**Note:** It is important to note that a primary consideration of the `ndtensor` framework is ensuring compatibility in two key areas:

- `autodiff`: the upcoming feature enabling rust to natively support automatic differentiation.
- [`ndarray`](https://docs.rs/ndarray): The crate is designed around the `ndarray` crate, which provides a powerful N-dimensional array type for Rust

## Overview

### Goals

- Provide a flexible and extensible framework for building neural network models in Rust.
- Support both shallow and deep neural networks with a focus on modularity and reusability.
- Enable easy integration with other libraries and frameworks in the Rust ecosystem.

### Roadmap

- [ ] **v1**:
  - [ ] **`ParamsBase`**: Design a basic structure for storing model parameters.
  - [ ]  **Traits**: Create a set of traits for defining the basics of a neural network model.
    - `Forward` and `Backward`: traits defining forward and backward propagation
    - `Model`: A trait for defining a neural network model.
    - `Predict`: A trait extending the basic [`Forward`](cnc::Forward) pass.
    - `Train`: A trait for training a neural network model.
- [ ] **v2**:
  - [ ] **Models**:
    - `Trainer`: A generic model trainer that can be used to train any model.
  - [ ] **Layers**: Implement a standard model configuration and parameters.
    - `LayerBase`: _functional_ wrappers for the `ParamsBase` structure.

## Usage

### Adding to your project

To use `ndtensor` in your project, add the following to your `Cargo.toml`:

```toml
[dependencies.ndtensor]
features = ["full"]
version = "0.1.x"
```

### Examples

#### **Example (1)**: Basic Usage

```rust

```

## Getting Started

### Prerequisites

To use `ndtensor`, you need to have the following installed:

- [Rust](https://www.rust-lang.org/tools/install) (version 1.85 or later)

### Installation

You can install the `rustup` toolchain using the following command:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

After installing `rustup`, you can install the latest stable version of Rust with:

```bash
rustup install stable
```

You can also install the latest nightly version of Rust with:

```bash
rustup install nightly
```

### Building from the source

Start by cloning the repository

```bash
git clone https://github.com/FL03/ndtensor.git
```

Then, navigate to the `ndtensor` directory:

```bash
cd ndtensor
```

#### _Using the `cargo` tool_

To build the crate, you can use the `cargo` tool. The following command will build the crate with all features enabled:

```bash
cargo build -r --locked --workspace --features full
```

To run the tests, you can use the following command:

```bash
cargo test -r --locked --workspace --features full
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

- [Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)
- [MIT](https://choosealicense.com/licenses/mit/)
