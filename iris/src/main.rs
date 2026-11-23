use std::time::Instant;

use anyhow::Context;
use clap::{Parser, ValueEnum};
use tutor_core::{data::DataSheet, designer::ModelDesigner, model::Model};

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
            // split rows alternately into training and test sets
            let mut train = DataSheet::from_schema(data_set.schema().clone());
            let mut test = DataSheet::from_schema(data_set.schema().clone());
            for (i, row) in data_set.iter().enumerate() {
                if i % 2 == 0 { train.push(row)?; } else { test.push(row)?; }
            }

            // setup a model fitted to the training rows and a learning rate
            let design = args
                .data_set
                .design_model()?;
            let design_str = design.serialize()?;
            let mut model = ModelDesigner::deserialize(&design_str)?
                .build(&train)
                .context("failed to build the model design")?;
            model.train_until_stop_improvement(&train, 0.5, 1e-4, 50)?;
            let test_len = test.iter().count();

            let start_time = Instant::now();
            let accuracy = model.est_accuracy(&test)?;
            let duration = start_time.elapsed();

            println!("Final accuracy: {:.5}%", accuracy * 100.0);
            println!(
                "Took {} ns to run {} predictions or {} ns/prediction",
                duration.as_nanos(), test_len,
                duration.as_nanos() / test_len as u128
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

            let data_len = data_set.iter().count();
            let start_time = Instant::now();
            let accuracy = model.est_accuracy(&data_set)?;
            let duration = start_time.elapsed();

            println!("Accuracy: {:.5}%", accuracy * 100.0);
            println!(
                "Took {} ns to run {} predictions or {} ns/prediction",
                duration.as_nanos(), data_len,
                duration.as_nanos() / data_len as u128
            );
        }
    }

    Ok(())
}
