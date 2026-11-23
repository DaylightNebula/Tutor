use anyhow::Context;
use tutor_core::{
    data::{DataSheet, DataType, Role, Schema},
    designer::ModelDesigner,
    layers::{input_schema::InputSchema, output_schema::OutputSchema, weights::Weights, zero_neg::ZeroNegative},
    model::Model
};

/// The columns of the census data sheet: every person's details as integer or enum inputs, and
/// their `income` band as an enum output.  `fnlwgt` is left out, as it is a sampling weight for
/// the census rather than a detail of the person.
pub(crate) fn schema() -> Schema {
    let mut schema = Schema::new();
    schema.push("age", Role::Input, DataType::Integer);
    schema.push("workclass", Role::Input, DataType::Enum);
    schema.push("education", Role::Input, DataType::Enum);
    schema.push("education.num", Role::Input, DataType::Integer);
    schema.push("marital.status", Role::Input, DataType::Enum);
    schema.push("occupation", Role::Input, DataType::Enum);
    schema.push("relationship", Role::Input, DataType::Enum);
    schema.push("race", Role::Input, DataType::Enum);
    schema.push("sex", Role::Input, DataType::Enum);
    schema.push("capital.gain", Role::Input, DataType::Integer);
    schema.push("capital.loss", Role::Input, DataType::Integer);
    schema.push("hours.per.week", Role::Input, DataType::Integer);
    schema.push("native.country", Role::Input, DataType::Enum);
    schema.push("income", Role::Output, DataType::Enum);
    return schema;
}

/// Loads the census data set from `./census.csv`.
///
/// Rows containing a `?`, which the data set uses for a missing value, are skipped.
///
/// # Errors
/// * The csv file cannot be read.
/// * A row is missing a column or has a value that does not match its column's type.
pub(crate) fn load_data() -> anyhow::Result<DataSheet> {
    let csv_content = std::fs::read_to_string("./census.csv")
        .context("failed to read ./census.csv")?;
    let csv_content = csv_content
        .lines()
        .filter(|line| !line.contains('?'))
        .collect::<Vec<_>>()
        .join("\n");

    let schema = schema();
    let data_sheet = DataSheet::from_csv(schema, &csv_content)?;
    return Ok(data_sheet);
}

/// Builds an untrained model shaped for this data set.
///
/// Uses a single 32 wide ReLU hidden layer.
///
/// # Parameters
/// * num_outputs: Number of income bands the model chooses between.
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
