use std::{collections::HashMap, error::Error, sync::LazyLock};

use clap::{Parser, Subcommand};
use llms_from_scratch_burn::{Listing, listings::ch02::L2_1};

static LISTINGS: LazyLock<HashMap<&str, Box<dyn Listing>>> = LazyLock::new(|| {
    let mut listings: HashMap<&str, Box<dyn Listing>> = HashMap::new();
    listings.insert("2.1", Box::new(L2_1));
    listings
});

#[derive(Debug, Parser)]
#[command(bin_name = "llms-from-scratch-burn")]
#[command(about = "A CLI for running examples and exercises.", long_about = None)]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Debug, Subcommand)]
enum Commands {
    /// Run examples
    Example {
        /// The example to run
        id: String,
    },
    /// Run exercises
    Exercise {
        /// The exercise to run
        id: String,
    },
    Listing {
        /// The listing to run
        id: String,
    },
}

fn main() -> Result<(), Box<dyn Error>> {
    let cli = Cli::parse();

    match cli.command {
        Commands::Example { id } => {
            println!("example: {:?}", id);
        }
        Commands::Exercise { id } => {
            println!("exercise: {:?}", id);
        }
        Commands::Listing { id } => {
            let listing = LISTINGS.get(id.as_str()).unwrap();
            listing.main()?;
        }
    }
    Ok(())
}
