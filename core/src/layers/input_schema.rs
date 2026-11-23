use anyhow::Context;
use getset::{CopyGetters, Getters};
use ndarray::{Array2, ArrayView2};
use serde::{Deserialize, Serialize};

use crate::{data::{DataConverter, DataSheet, DataStorage}, layers::Layer};


/// Turns a row of a [`DataSheet`] into the numbers the rest of the model works with.  It must be
/// the first layer of a model.  It replaces a separate scalars layer: numeric columns are
/// normalized here, and enum columns are one-hot encoded so the model never treats an option's
/// position as a size.
#[derive(Getters, Clone, Debug, Default, Serialize, Deserialize)]
pub struct InputSchema {
    #[getset(get = "pub")]
    columns: Option<Vec<SchemaColumn>>
}

/// One data sheet column read or written by a schema layer.
///
/// Fields:
/// * index: Position of the column in each data sheet row.
/// * name: Name of the column in the sheet's schema.
/// * converter: Converter fitted to the column's sample values.
#[derive(Getters, CopyGetters, Clone, Debug, Serialize, Deserialize)]
pub struct SchemaColumn {
    #[getset(get_copy = "pub")]
    index: usize,
    #[getset(get = "pub")]
    name: String,
    #[getset(get = "pub")]
    converter: DataConverter
}

impl SchemaColumn {
    /// Pairs each converter from one of the data sheet build functions with its column index.
    pub fn from_converters(
        sheet: &DataSheet,
        converters: Vec<(String, DataConverter)>
    ) -> anyhow::Result<Vec<Self>> {
        converters
            .into_iter()
            .map(|(name, converter)| Ok(Self { index: sheet.column_idx(name.as_str())?, name, converter }))
            .collect()
    }
}

impl InputSchema {
    /// Creates an input schema layer that is fitted to its columns when the model is built.
    pub fn new() -> Self { Self::default() }
}

#[typetag::serde]
impl Layer for InputSchema {

    /// Number of encoded input weights, or the previous width until the schema is prepared.
    fn output_size(&self, previous_layer_num_outputs: usize) -> usize {
        match &self.columns {
            Some(columns) => columns.iter()
                .map(|column| column.converter.num_weights())
                .sum(),
            None => previous_layer_num_outputs
        }
    }

    /// Fits a converter to every input column of the sample sheet.
    fn prepare_train(
        &mut self,
        sample_data: &DataSheet,
        _previous_layer_num_outputs: usize
    ) -> anyhow::Result<()> {
        let converters = sample_data
            .build_input_schema()
            .context("failed to build the input schema")?;
        anyhow::ensure!(!converters.is_empty(), "the data sheet schema has no input columns");

        self.columns = Some(SchemaColumn::from_converters(sample_data, converters)?);
        Ok(())
    }

    /// Passes the already encoded inputs through unchanged.
    fn transform(&self, inputs: ArrayView2<f32>) -> anyhow::Result<Array2<f32>> {
        Ok(inputs.to_owned())
    }

    /// Passes the gradients back unchanged, nothing in this layer is learned.
    fn back_propagate(
        &mut self,
        _inputs: ArrayView2<f32>,
        _outputs: ArrayView2<f32>,
        output_gradient: ArrayView2<f32>
    ) -> anyhow::Result<Array2<f32>> {
        Ok(output_gradient.to_owned())
    }

    /// Encodes every input column of each row, joined in column index order, into one batch row
    /// per data sheet row.
    fn encode_inputs(&self, rows: &[&[DataStorage]]) -> anyhow::Result<Array2<f32>> {
        let Some(columns) = &self.columns else { anyhow::bail!("InputSchema layer not prepared") };
        encode_columns(columns, rows, "input")
    }
}

/// Encodes the given columns of every row into one flat batch, one batch row per data sheet row
/// with the columns joined in order.  Shared by the input and output schemas.
pub(crate) fn encode_columns(
    columns: &[SchemaColumn],
    rows: &[&[DataStorage]],
    kind: &str
) -> anyhow::Result<Array2<f32>> {
    let width = columns
        .iter()
        .map(|column| column.converter.num_weights())
        .sum::<usize>();

    // encode every row back to back into one flat buffer
    let mut encoded = Vec::with_capacity(width * rows.len());
    for row in rows {
        for column in columns {
            let value = row
                .get(column.index)
                .with_context(|| format!("row has no value for {kind} column {}", column.name))?;
            column.converter
                .encode(value, &mut encoded)
                .with_context(|| format!("failed to encode {kind} column {}", column.name))?;
        }
    }

    Array2::from_shape_vec((rows.len(), width), encoded)
        .with_context(|| format!("encoded {kind} columns did not fill a {} × {width} batch", rows.len()))
}
