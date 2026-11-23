use serde::Deserialize;
use tutor_core::{
    data::{DataSheet, DataType, Role, Schema},
    designer::ModelDesigner,
    layers::{input_schema::InputSchema, output_schema::OutputSchema, weights::Weights},
    model::Model
};

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

pub(crate) fn schema() -> Schema {
    let mut schema = Schema::new();
    schema.push("SepalLengthCm", Role::Input, DataType::Decimal);
    schema.push("SepalWidthCm", Role::Input, DataType::Decimal);
    schema.push("PetalLengthCm", Role::Input, DataType::Decimal);
    schema.push("PetalWidthCm", Role::Input, DataType::Decimal);
    schema.push("Species", Role::Output, DataType::Enum);
    return schema;
}

pub(crate) fn load_data() -> anyhow::Result<DataSheet> {
    let schema = schema();
    let csv_content = std::fs::read_to_string("./Iris.csv")?;
    let data_sheet = DataSheet::from_csv(schema, &csv_content)?;
    return Ok(data_sheet);
}

pub(crate) fn design_model(num_outputs: usize) -> anyhow::Result<ModelDesigner> {
    Model::designer()
        .add_layer(InputSchema::new())?
        .add_layer(Weights::random(num_outputs, 1e-2))?
        .add_layer(OutputSchema::new())
}
