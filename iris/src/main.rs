use std::time::Instant;

use clap::{Parser, ValueEnum};
use serde::Deserialize;

use crate::{layers::{no_neg::NoNegative, scalars::Scalars, softmax::SoftMax, weights::Weights}, model::Model};

pub mod layers;
pub mod model;

#[derive(Parser, Debug)]
#[command(version, about, long_about = None)]
pub struct Args {
    #[arg(long, default_value = "train")]
    mode: Mode,
    #[arg(short, long, default_value = "./Iris.csv")]
    input: String,
    #[arg(short, long, default_value = "./model.tutor")]
    model: String
}

#[derive(Clone, Copy, ValueEnum, Debug, PartialEq, Eq, PartialOrd, Ord)]
pub enum Mode {
    Train,
    Execute
}

#[derive(Debug, Deserialize)]
pub struct Flower {
    #[serde(rename = "SepalLengthCm")]
    pub s_length: f32,
    #[serde(rename = "SepalWidthCm")]
    pub s_width: f32,
    #[serde(rename = "PetalLengthCm")]
    pub p_length: f32,
    #[serde(rename = "PetalWidthCm")]
    pub p_width: f32,
    #[serde(rename = "Species")]
    pub species: Species
}

#[derive(Clone, Copy, Debug, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum Species {
    Setosa = 0,
    Versicolor = 1,
    Virginica = 2
}

impl Flower {
    pub fn features(&self) -> [f32; 4] {
        [self.s_length, self.s_width, self.p_length, self.p_width]
    }
}

fn main() -> anyhow::Result<()> {
    let args = Args::parse();

    // load all rows, split 50/50 between a training and a test set
    let mut csv_reader = csv::Reader::from_path(args.input)?;
    let flowers = csv_reader.deserialize()
        .collect::<Result<Vec<Flower>, _>>()?
        .into_iter()
        .map(|flower| (Box::new(flower.features()) as Box<[f32]>, flower.species as usize))
        .collect::<Vec<_>>();

    match args.mode {
        Mode::Train => {
            // split flowers into training and test sets
            let mut flip_flip = false;
            let (train, test): (Vec<_>, Vec<_>) = flowers
                .iter()
                .cloned()
                .partition(|_| {
                    flip_flip = !flip_flip;
                    flip_flip
                });

            // setup a model and learning rate
            let scalars = train.iter().map(|a| a.0.to_vec()).collect::<Vec<_>>();
            let mut model = Model::blank(4, 3)
                .add_layer(Scalars::from_fit(&scalars))?
                .add_layer(Weights::random(4, 8, 1e-2))?
                .add_layer(NoNegative::new(8))?
                .add_layer(Weights::random(8, 3, 1e-2))?
                .add_layer(SoftMax::new(3))?;
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
