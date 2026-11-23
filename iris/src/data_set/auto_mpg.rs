use anyhow::Context;
use tutor_core::{
    data::{DataSheet, DataType, Role, Schema},
    designer::ModelDesigner,
    layers::{input_schema::InputSchema, output_schema::OutputSchema, weights::Weights, zero_neg::ZeroNegative},
    model::Model
};

/// The columns of the auto mpg data sheet: every car spec as a decimal input, and the car's fuel
/// efficiency band as an enum output.
pub(crate) fn schema() -> Schema {
    let mut schema = Schema::new();
    schema.push("cylinders", Role::Input, DataType::Decimal);
    schema.push("displacement", Role::Input, DataType::Decimal);
    schema.push("horsepower", Role::Input, DataType::Decimal);
    schema.push("weight", Role::Input, DataType::Decimal);
    schema.push("acceleration", Role::Input, DataType::Decimal);
    schema.push("model year", Role::Input, DataType::Decimal);
    schema.push("origin", Role::Input, DataType::Decimal);
    schema.push("car name", Role::Output, DataType::Enum);
    return schema;
}

/// Loads the auto mpg data set from `./auto-mpg.csv`.
pub(crate) fn load_data() -> anyhow::Result<DataSheet> {
    let csv_content = std::fs::read_to_string("./auto-mpg.csv")
        .context("failed to read ./auto-mpg.csv")?;
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
pub(crate) fn design_model(num_outputs: usize) -> anyhow::Result<ModelDesigner> {
    Model::designer()
        .add_layer(InputSchema::new())?
        .add_layer(Weights::random(64, 1e-2))?
        .add_layer(ZeroNegative)?
        .add_layer(Weights::random(num_outputs, 1e-2))?
        .add_layer(OutputSchema::new())
}
