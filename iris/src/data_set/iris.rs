use serde::Deserialize;
use tutor_core::{layers::{scalars::Scalars, softmax::SoftMax, weights::Weights}, model::Model};

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

pub(crate) fn load_data() -> anyhow::Result<Vec<(Box<[f32]>, usize)>> {
    let mut csv_reader = csv::Reader::from_path("./Iris.csv")?;
    return Ok(
        csv_reader.deserialize()
            .collect::<Result<Vec<Flower>, _>>()?
            .into_iter()
            .map(|flower| (Box::new(flower.features()) as Box<[f32]>, flower.species as usize))
            .collect::<Vec<_>>()
    );
}

pub(crate) fn create_model(scalars: Scalars, num_inputs: usize, num_outputs: usize) -> anyhow::Result<Model> {
    Model::blank(num_inputs, num_outputs)
        .add_layer(scalars)?
        .add_layer(Weights::random(num_inputs, num_outputs, 1e-2))?
        .add_layer(SoftMax::new(num_outputs))
}
