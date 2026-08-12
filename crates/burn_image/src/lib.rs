//! Model-neutral contracts for image generation and editing runtimes.
//!
//! This crate intentionally has no dependency on Burn tensor backends or a
//! concrete model implementation. [`ImageModel`] uses an associated output type
//! so an implementation can keep results device-resident.

pub mod artifacts;
pub mod capabilities;
pub mod error;
pub mod runtime;
pub mod types;

pub use artifacts::*;
pub use capabilities::*;
pub use error::*;
pub use runtime::*;
pub use types::*;
