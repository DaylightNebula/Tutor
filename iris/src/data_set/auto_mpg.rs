use anyhow::Context;
use serde::Deserialize;
use tutor_core::{
    data::{DataSheet, DataStorage, DataType, Role, Schema},
    designer::ModelDesigner,
    layers::{input_schema::InputSchema, output_schema::OutputSchema, weights::Weights, zero_neg::ZeroNegative},
    model::Model
};

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

/// Names of the input columns, in the same order as `Car::features`.
const INPUT_COLUMNS: [&str; 7] = [
    "cylinders",
    "displacement",
    "horsepower",
    "weight",
    "acceleration",
    "model year",
    "origin"
];

/// Name of the output column holding each car's `MpgClass`.
const CLASS_COLUMN: &str = "mpg_class";

/// The columns of the auto mpg data sheet: every car spec as a decimal input, and the car's
/// `MpgClass` as an enum output.
pub(crate) fn schema() -> Schema {
    let mut schema = Schema::new();
    for name in INPUT_COLUMNS {
        schema.push(name, Role::Input, DataType::Decimal);
    }
    schema.push(CLASS_COLUMN, Role::Output, DataType::Enum);
    return schema;
}

/// Loads the auto mpg data set from `./auto-mpg.csv`.
///
/// Each car becomes a sheet row of its specs plus its `MpgClass`.  The class is worked out from
/// the car's mpg, so the csv is parsed as `Car`s and pushed into the sheet row by row instead of
/// with `DataSheet::from_csv`.  Rows that are missing a horsepower value are skipped.
///
/// # Errors
/// * The csv file cannot be opened.
/// * A row cannot be parsed as a `Car`.
pub(crate) fn load_data() -> anyhow::Result<DataSheet> {
    let mut csv_reader = csv::Reader::from_path("./auto-mpg.csv")
        .context("failed to open ./auto-mpg.csv")?;

    // parse every car, dropping the ones with missing features
    let cars = csv_reader
        .deserialize()
        .collect::<Result<Vec<Car>, _>>()
        .context("failed to parse auto-mpg.csv")?;

    // lay each car out in the sheet's column order
    let mut sheet = DataSheet::from_schema(schema());
    for car in cars {
        let Some(features) = car.features() else { continue };
        let row = (0 .. sheet.columns().len())
            .map(|idx| {
                let name = sheet.column_name(idx)?;
                if name == CLASS_COLUMN {
                    return Ok(DataStorage::Enum(format!("{:?}", MpgClass::from_mpg(car.mpg))));
                }
                let position = INPUT_COLUMNS
                    .iter()
                    .position(|column| *column == name)
                    .with_context(|| format!("auto mpg schema has an unknown column {name}"))?;
                Ok(DataStorage::Decimal(features[position]))
            })
            .collect::<anyhow::Result<Vec<_>>>()?;
        sheet.push(&row)?;
    }

    Ok(sheet)
}

/// Builds an untrained model shaped for this data set.
///
/// Uses a single 64 wide ReLU hidden layer.  The fuel bands are not linearly separable from the
/// car specs, and accuracy rose with width up to about 64 then leveled off, while a second hidden
/// layer lowered it.
///
/// # Parameters
/// * num_outputs: Number of classes the model chooses between.
///
/// # Errors
/// * A layer cannot be added to the design.
pub(crate) fn design_model(num_outputs: usize) -> anyhow::Result<ModelDesigner> {
    Model::designer()
        .add_layer(InputSchema::new())?
        .add_layer(Weights::random(64, 1e-2))?
        .add_layer(ZeroNegative)?
        .add_layer(Weights::random(num_outputs, 1e-2))?
        .add_layer(OutputSchema::new())
}
