use tutor_core::{
    data::{DataSheet, DataType, Role, Schema},
    designer::ModelDesigner,
    layers::{input_schema::InputSchema, output_schema::OutputSchema, weights::Weights, zero_neg::ZeroNegative},
    model::Model
};

/// Number of diagnoses, benign (`B`) or malignant (`M`), and so the number of outputs a model
/// needs for this set.
pub const NUM_CLASSES: usize = 2;

/// Names of the 30 tumor measurement columns, the mean, standard error, and worst value of ten
/// cell nucleus measurements.
const MEASUREMENT_COLUMNS: [&str; 30] = [
    "radius_mean", "texture_mean", "perimeter_mean", "area_mean", "smoothness_mean",
    "compactness_mean", "concavity_mean", "concave points_mean", "symmetry_mean",
    "fractal_dimension_mean",
    "radius_se", "texture_se", "perimeter_se", "area_se", "smoothness_se",
    "compactness_se", "concavity_se", "concave points_se", "symmetry_se",
    "fractal_dimension_se",
    "radius_worst", "texture_worst", "perimeter_worst", "area_worst", "smoothness_worst",
    "compactness_worst", "concavity_worst", "concave points_worst", "symmetry_worst",
    "fractal_dimension_worst"
];

/// The columns of the breast cancer data sheet: every measurement as a decimal input, and the
/// `diagnosis` code as an enum output.
pub(crate) fn schema() -> Schema {
    let mut schema = Schema::new();
    for name in MEASUREMENT_COLUMNS {
        schema.push(name, Role::Input, DataType::Decimal);
    }
    schema.push("diagnosis", Role::Output, DataType::Enum);
    return schema;
}

/// Loads the breast cancer data set from `./breast_cancer.csv`.
///
/// # Errors
/// * The csv file cannot be read.
/// * A row is missing a column or has a measurement that is not a number.
pub(crate) fn load_data() -> anyhow::Result<DataSheet> {
    let csv_content = std::fs::read_to_string("./breast_cancer.csv")?;
    let data_sheet = DataSheet::from_csv(schema(), &csv_content)?;
    return Ok(data_sheet);
}

/// Builds an untrained model shaped for this data set.
///
/// Uses a single 32 wide ReLU hidden layer.  The diagnoses are close to linearly separable, so a
/// linear model is nearly as good, but a 32 wide hidden layer gave a small consistent gain that
/// stopped growing past that width.  Deeper stacks did not help.
///
/// # Parameters
/// * num_outputs: Number of classes the model chooses between.
///
/// # Errors
/// * A layer cannot be added to the design.
pub(crate) fn design_model(num_outputs: usize) -> anyhow::Result<ModelDesigner> {
    Model::designer()
        .add_layer(InputSchema::new())?
        .add_layer(Weights::random(32, 1e-2))?
        .add_layer(ZeroNegative)?
        .add_layer(Weights::random(num_outputs, 1e-2))?
        .add_layer(OutputSchema::new())
}
