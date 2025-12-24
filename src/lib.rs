use std::error::Error;

pub mod listings;

pub trait Listing: Send + Sync {
    fn main(&self) -> Result<(), Box<dyn Error>>;
}
