use std::time::Instant;

use anyhow::Context;
use clap::{Parser, ValueEnum};

use crate::{data_set::DataSet, layers::{no_neg::NoNegative, scalars::Scalars, softmax::SoftMax, weights::Weights}, model::Model};

pub mod data_set;
pub mod layers;
pub mod model;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    #[arg(long, default_value = "train")]
    mode: Mode,
    #[arg(short, long, default_value = "iris")]
    data_set: DataSet,
    #[arg(short, long, default_value = "./model.tutor")]
    model: String
}

#[derive(Clone, Copy, ValueEnum, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mode {
    Train,
    Execute
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    let data_set = args.data_set.load_data()?;

    match args.mode {
        Mode::Train => {
            // split flowers into training and test sets
            let mut flip_flip = false;
            let (train, test): (Vec<_>, Vec<_>) = data_set
                .iter()
                .cloned()
                .partition(|_| {
                    flip_flip = !flip_flip;
                    flip_flip
                });

            // setup a model sized to the data set and a learning rate
            let num_inputs = train
                .first()
                .context("data set has no rows to train on")?
                .0
                .len();
            let num_outputs = args.data_set.num_classes();
            let scalars = train.iter().map(|a| a.0.to_vec()).collect::<Vec<_>>();
            let mut model = Model::blank(num_inputs, num_outputs)
                .add_layer(Scalars::from_fit(&scalars))?
                .add_layer(Weights::random(num_inputs, 8, 1e-2))?
                .add_layer(NoNegative::new(8))?
                .add_layer(Weights::random(8, num_outputs, 1e-2))?
                .add_layer(SoftMax::new(num_outputs))?;
            model.train_until_stop_improvement(&train, 5e-1, 1e-4, 50)?;

            let start_time = Instant::now();
            let accuracy = model.est_accuracy(&test);
            let duration = start_time.elapsed();

            println!("Final accuracy: {:.5}%", accuracy * 100.0);
            println!(
                "Took {} ns to run {} predictions or {} ns/prediction",
                duration.as_nanos(), test.len(),
                duration.as_nanos() / test.len() as u128
            );
        },
        Mode::Execute => {
            todo!("Execute")
        }
    }

    Ok(())
}
