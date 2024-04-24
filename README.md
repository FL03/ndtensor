# ndtensor

[![Clippy](https://github.com/FL03/ndtensor/actions/workflows/clippy.yml/badge.svg)](https://github.com/FL03/ndtensor/actions/workflows/clippy.yml)
[![Rust](https://github.com/FL03/ndtensor/actions/workflows/rust.yml/badge.svg)](https://github.com/FL03/ndtensor/actions/workflows/rust.yml)

[![crates.io](https://img.shields.io/crates/v/ndtensor.svg)](https://crates.io/crates/ndtensor)
[![docs.rs](https://docs.rs/ndtensor/badge.svg)](https://docs.rs/ndtensor)

***

Welcome to ndtensor, a Rust library for n-dimensional tensors designed for flexibility and performance.


## Getting Started

### Building from the source

#### _Clone the repository_

```bash
git clone https://github.com/FL03/ndtensor
```

### Usage

```rust
extern crate ndtensor;

use ndtensor::Tensor;

fn main() -> Result<(), Box<dyn std::error::Error> {
    let shape = (3, 3);
    
    let tensor = Tensor::linspace(0f64, 8f64, 9).into_shape(shape)?;
    println!("{:?}", tensor);

    Ok(())
}
```

## Contributing

Pull requests are welcome. For major changes, please open an issue first
to discuss what you would like to change.

Please make sure to update tests as appropriate.

## License

- [Apache-2.0](https://choosealicense.com/licenses/apache-2.0/)
