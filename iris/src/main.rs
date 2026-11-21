use std::time::Instant;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use tutor_core::{layers::scalars::Scalars, model::Model};

use crate::data_set::DataSet;

pub mod data_set;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    #[arg(long, default_value = "train")]
    mode: Mode,
    #[arg(short, long, default_value = "iris")]
    data_set: DataSet,
    #[arg(short, long)]
    model: Option<String>
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
            let mut model = args
                .data_set
                .create_model(Scalars::from_fit(&scalars), num_inputs, num_outputs)?;
            model.train_until_stop_improvement(&train, 0.5, 1e-4, 50)?;

            let start_time = Instant::now();
            let accuracy = model.est_accuracy(&test);
            let duration = start_time.elapsed();

            println!("Final accuracy: {:.5}%", accuracy * 100.0);
            println!(
                "Took {} ns to run {} predictions or {} ns/prediction",
                duration.as_nanos(), test.len(),
                duration.as_nanos() / test.len() as u128
            );

            // save the trained model, skipped when no path is given such as for test runs
            if let Some(model_path) = args.model {
                let binary_model = postcard::to_allocvec(&model)?;
                std::fs::write(&model_path, binary_model)
                    .with_context(|| format!("failed to write model to {model_path}"))?;
            }
        },
        Mode::Execute => {
            let model_path = args
                .model
                .context("--model is required to execute a saved model")?;
            let binary_model = std::fs::read(&model_path)
                .with_context(|| format!("failed to read model from {model_path}"))?;
            let model: Model = postcard::from_bytes(&binary_model)?;

            let start_time = Instant::now();
            let accuracy = model.est_accuracy(&data_set);
            let duration = start_time.elapsed();

            println!("Accuracy: {:.5}%", accuracy * 100.0);
            println!(
                "Took {} ns to run {} predictions or {} ns/prediction",
                duration.as_nanos(), data_set.len(),
                duration.as_nanos() / data_set.len() as u128
            );
        }
    }

    Ok(())
}
