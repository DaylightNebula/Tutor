use anyhow::Context;
use serde::Deserialize;
use tutor_core::{layers::{scalars::Scalars, softmax::SoftMax, weights::Weights, zero_neg::ZeroNegative}, model::Model};

/// A single car row from the auto mpg data set.
///
/// Fields:
/// * mpg: Miles per gallon, used to pick the car's `MpgClass`.
/// * cylinders: Number of engine cylinders.
/// * displacement: Engine displacement in cubic inches.
/// * horsepower: Engine horsepower, `None` when the data set lists it as `?`.
/// * weight: Vehicle weight in pounds.
/// * acceleration: Seconds taken to reach 60 mph.
/// * model_year: Two digit model year.
/// * origin: Region of origin, 1 for USA, 2 for Europe, 3 for Japan.
#[derive(Debug, Deserialize)]
pub struct Car {
    pub mpg: f32,
    pub cylinders: f32,
    pub displacement: f32,
    #[serde(deserialize_with = "csv::invalid_option")]
    pub horsepower: Option<f32>,
    pub weight: f32,
    pub acceleration: f32,
    #[serde(rename = "model year")]
    pub model_year: f32,
    pub origin: f32
}

/// The fuel efficiency band a car falls into, used as the classification target since the model
/// predicts classes rather than continuous values.
///
/// Variants:
/// * Low: Less than 20 mpg.
/// * Medium: At least 20 mpg but less than 27 mpg.
/// * High: 27 mpg or more.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord)]
#[repr(usize)]
pub enum MpgClass {
    Low = 0,
    Medium = 1,
    High = 2
}

/// Number of variants in `MpgClass`, and so the number of outputs a model needs for this set.
pub const NUM_CLASSES: usize = 3;

impl MpgClass {
    /// Buckets a raw mpg value into its efficiency band.
    ///
    /// # Parameters
    /// * mpg: Miles per gallon of the car.
    pub fn from_mpg(mpg: f32) -> Self {
        if mpg < 20.0 {
            MpgClass::Low
        } else if mpg < 27.0 {
            MpgClass::Medium
        } else {
            MpgClass::High
        }
    }
}

impl Car {
    /// Gives the model's input features for this car, or `None` if any feature is missing.
    ///
    /// The `car name` column and `mpg` are left out, as the name is unique per car and the mpg is
    /// what is being predicted.
    pub fn features(&self) -> Option<[f32; 7]> {
        Some([
            self.cylinders,
            self.displacement,
            self.horsepower?,
            self.weight,
            self.acceleration,
            self.model_year,
            self.origin
        ])
    }
}

/// Loads the auto mpg data set from `./auto-mpg.csv`.
///
/// Each row becomes its feature list paired with the index of its `MpgClass`.  Rows that are
/// missing a horsepower value are skipped.
///
/// # Errors
/// * The csv file cannot be opened.
/// * A row cannot be parsed as a `Car`.
pub(crate) fn load_data() -> anyhow::Result<Vec<(Box<[f32]>, usize)>> {
    let mut csv_reader = csv::Reader::from_path("./auto-mpg.csv")
        .context("failed to open ./auto-mpg.csv")?;

    // parse every car, dropping the ones with missing features
    let cars = csv_reader
        .deserialize()
        .collect::<Result<Vec<Car>, _>>()
        .context("failed to parse auto-mpg.csv")?;
    Ok(
        cars
            .into_iter()
            .filter_map(|car| {
                let features = car.features()?;
                Some((Box::new(features) as Box<[f32]>, MpgClass::from_mpg(car.mpg) as usize))
            })
            .collect()
    )
}

/// Width of the hidden layer in the model built by `create_model`.
const HIDDEN_WIDTH: usize = 64;

/// Builds an untrained model shaped for this data set.
///
/// Uses a single 64 wide ReLU hidden layer.  The fuel bands are not linearly separable from the
/// car specs, and accuracy rose with width up to about 64 then leveled off, while a second hidden
/// layer lowered it.
///
/// # Parameters
/// * scalars: Input normalization fitted to the training rows, used as the first layer.
/// * num_inputs: Number of input features in each row.
/// * num_outputs: Number of classes the model chooses between.
///
/// # Errors
/// * `scalars` does not take `num_inputs` inputs, so the layers do not line up.
pub(crate) fn create_model(
    sample_data: &[(Box<[f32]>, usize)],
    num_inputs: usize,
    num_outputs: usize
) -> anyhow::Result<Model> {
    Model::builder(num_inputs, num_outputs)
        .add_layer(Scalars::new(num_inputs))?
        .add_layer(Weights::random(num_inputs, HIDDEN_WIDTH, 1e-2))?
        .add_layer(ZeroNegative::new(HIDDEN_WIDTH))?
        .add_layer(Weights::random(HIDDEN_WIDTH, num_outputs, 1e-2))?
        .add_layer(SoftMax::new(num_outputs))?
        .build(sample_data)
}
