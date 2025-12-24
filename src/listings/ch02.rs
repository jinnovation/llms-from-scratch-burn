use std::error::Error;

use crate::Listing;

pub struct L2_1;

impl Listing for L2_1 {
    fn main(&self) -> Result<(), Box<dyn Error>> {
        println!("listing 2.1");
        Ok(())
    }
}
