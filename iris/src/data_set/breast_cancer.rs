use tutor_core::{
    data::{DataSheet, DataType, Role, Schema},
    designer::ModelDesigner,
    layers::{input_schema::InputSchema, output_schema::OutputSchema, weights::Weights, zero_neg::ZeroNegative},
    model::Model
};

/// The columns of the breast cancer data sheet: every measurement as a decimal input, and the
/// `diagnosis` code as an enum output.
pub(crate) fn schema() -> Schema {
    let mut schema = Schema::new();
    schema.push("radius_mean", Role::Input, DataType::Decimal);
    schema.push("texture_mean", Role::Input, DataType::Decimal);
    schema.push("perimeter_mean", Role::Input, DataType::Decimal);
    schema.push("area_mean", Role::Input, DataType::Decimal);
    schema.push("smoothness_mean", Role::Input, DataType::Decimal);
    schema.push("compactness_mean", Role::Input, DataType::Decimal);
    schema.push("concavity_mean", Role::Input, DataType::Decimal);
    schema.push("concave points_mean", Role::Input, DataType::Decimal);
    schema.push("symmetry_mean", Role::Input, DataType::Decimal);
    schema.push("fractal_dimension_mean", Role::Input, DataType::Decimal);
    schema.push("radius_se", Role::Input, DataType::Decimal);
    schema.push("texture_se", Role::Input, DataType::Decimal);
    schema.push("perimeter_se", Role::Input, DataType::Decimal);
    schema.push("area_se", Role::Input, DataType::Decimal);
    schema.push("smoothness_se", Role::Input, DataType::Decimal);
    schema.push("compactness_se", Role::Input, DataType::Decimal);
    schema.push("concavity_se", Role::Input, DataType::Decimal);
    schema.push("concave points_se", Role::Input, DataType::Decimal);
    schema.push("symmetry_se", Role::Input, DataType::Decimal);
    schema.push("fractal_dimension_se", Role::Input, DataType::Decimal);
    schema.push("radius_worst", Role::Input, DataType::Decimal);
    schema.push("texture_worst", Role::Input, DataType::Decimal);
    schema.push("perimeter_worst", Role::Input, DataType::Decimal);
    schema.push("area_worst", Role::Input, DataType::Decimal);
    schema.push("smoothness_worst", Role::Input, DataType::Decimal);
    schema.push("compactness_worst", Role::Input, DataType::Decimal);
    schema.push("concavity_worst", Role::Input, DataType::Decimal);
    schema.push("concave points_worst", Role::Input, DataType::Decimal);
    schema.push("symmetry_worst", Role::Input, DataType::Decimal);
    schema.push("fractal_dimension_worst", Role::Input, DataType::Decimal);
    schema.push("diagnosis", Role::Output, DataType::Enum);
    return schema;
}

/// Loads the breast cancer data set from `./breast_cancer.csv`.
pub(crate) fn load_data() -> anyhow::Result<DataSheet> {
    let csv_content = std::fs::read_to_string("./breast_cancer.csv")?;
    let data_sheet = DataSheet::from_csv(schema(), &csv_content)?;
    return Ok(data_sheet);
}

/// Builds an untrained model shaped for this data set.
pub(crate) fn design_model(num_outputs: usize) -> anyhow::Result<ModelDesigner> {
    Model::designer()
        .add_layer(InputSchema::new())?
        .add_layer(Weights::random(32, 1e-2))?
        .add_layer(ZeroNegative)?
        .add_layer(Weights::random(num_outputs, 1e-2))?
        .add_layer(OutputSchema::new())
}
