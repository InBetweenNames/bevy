//! Storage layouts for ECS data.

pub(super) mod aligned_vec;
mod blob_vec;
mod resource;
mod sparse_set;
mod table;

pub use resource::*;
pub use sparse_set::*;
pub use table::*;

/// The raw data stores of a [World](crate::world::World)
#[derive(Default)]
pub struct Storages {
    pub sparse_sets: SparseSets,
    pub tables: Tables,
    pub resources: Resources,
}
