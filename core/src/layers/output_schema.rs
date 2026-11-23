use anyhow::Context;
use getset::Getters;
use serde::{Deserialize, Serialize};

use crate::{
    data::{DataConverter, DataSheet, DataStorage},
    layers::{Layer, input_schema::SchemaColumn}
};

/// Turns the raw scores of the layer before it into one probability per option of every output
/// column, and reads them back into [`DataSheet`] values.  It must be the last layer of a model,
/// and it replaces a separate softmax layer.  Only enum output columns are supported, so every
/// model built with it is a classifier.
#[derive(Getters, Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutputSchema {
    #[getset(get = "pub")]
    columns: Option<Vec<SchemaColumn>>
}

/// An output column, its enum options, and the range of this layer's weights that belong to it.
type ColumnSlice<'a> = (&'a SchemaColumn, &'a [String], std::ops::Range<usize>);

impl OutputSchema {
    /// Creates an output schema layer that is fitted to its columns when the model is built.
    pub fn new() -> Self { Self::default() }

    /// The prepared columns, each paired with its options and its slice of this layer's weights.
    fn column_slices(&self) -> anyhow::Result<Vec<ColumnSlice<'_>>> {
        let Some(columns) = &self.columns else { anyhow::bail!("OutputSchema layer not prepared") };

        let mut offset = 0;
        let mut slices = Vec::with_capacity(columns.len());
        for column in columns {
            let DataConverter::Enum { options } = column.converter() else {
                anyhow::bail!("output column {} is not an enum", column.name())
            };
            slices.push((column, options.as_slice(), offset .. offset + options.len()));
            offset += options.len();
        }

        Ok(slices)
    }
}

#[typetag::serde]
impl Layer for OutputSchema {

    /// One probability per option of every output column.  Before the schema is prepared the
    /// width is not known yet, so the previous layer's width is passed through.
    fn output_size(&self, previous_layer_num_outputs: usize) -> usize {
        match &self.columns {
            Some(columns) => columns.iter()
                .map(|column| column.converter().num_weights())
                .sum(),
            None => previous_layer_num_outputs
        }
    }

    /// Fits a converter to every output column of the sample sheet.
    fn prepare_train(
        &mut self,
        sample_data: &DataSheet,
        previous_layer_num_outputs: usize
    ) -> anyhow::Result<()> {
        let converters = sample_data
            .build_output_schema()
            .context("failed to build the output schema")?;
        anyhow::ensure!(!converters.is_empty(), "the data sheet schema has no output columns");
        self.columns = Some(SchemaColumn::from_converters(sample_data, converters)?);

        // every output must be an enum and the layer before must give one score per option
        self.column_slices()?;
        let width = self.output_size(previous_layer_num_outputs);
        anyhow::ensure!(
            width == previous_layer_num_outputs,
            "the output columns have {width} options in total but the layer before OutputSchema gives {previous_layer_num_outputs} scores"
        );

        Ok(())
    }

    /// Applies a softmax to the scores of each output column.
    fn transform(&self, scores: &[f32]) -> anyhow::Result<Box<[f32]>> {
        let mut probabilities = vec![0.0; scores.len()];
        for (_, _, range) in self.column_slices()? {
            let column_scores = &scores[range.clone()];
            let max = column_scores
                .iter()
                .copied()
                .fold(f32::NEG_INFINITY, f32::max);

            // exponentiate then divide by the column's total
            let exps = column_scores
                .iter()
                .map(|score| (score - max).exp())
                .collect::<Vec<_>>();
            let sum = exps.iter().sum::<f32>();
            for (probability, exp) in probabilities[range].iter_mut().zip(exps) {
                *probability = exp / sum;
            }
        }

        Ok(probabilities.into_boxed_slice())
    }

    /// Passes the gradients back through each column's softmax.
    fn back_propagate(
        &mut self,
        _inputs: &[f32],
        outputs: &[f32],
        output_gradient: &[f32]
    ) -> anyhow::Result<Box<[f32]>> {
        let mut input_grads = vec![0.0; outputs.len()];
        for (_, _, range) in self.column_slices()? {
            let column_outputs = &outputs[range.clone()];
            let column_grads = &output_gradient[range.clone()];

            // Σ g·y appears in every input's gradient, so compute it once
            let dot = column_grads
                .iter()
                .zip(column_outputs)
                .map(|(g, y)| g * y)
                .sum::<f32>();

            // dL/dz_i = y_i · (g_i − Σ g·y)
            for ((input_grad, y), g) in input_grads[range].iter_mut().zip(column_outputs).zip(column_grads) {
                *input_grad = y * (g - dot);
            }
        }

        Ok(input_grads.into_boxed_slice())
    }

    /// Cross-entropy of each output column's probabilities against the row's option, summed over
    /// the columns, with its gradient.
    fn output_loss(&self, outputs: &[f32], row: &[DataStorage]) -> anyhow::Result<(f32, Box<[f32]>)> {
        let mut loss = 0.0;
        let mut grads = vec![0.0; outputs.len()];
        for (column, _, range) in self.column_slices()? {
            let Some(DataStorage::Enum(expected)) = row.get(column.index()) else {
                anyhow::bail!("row has no enum value for output column {}", column.name())
            };
            let target = range.start + column.converter()
                .option_index(expected)
                .with_context(|| format!("unknown option in output column {}", column.name()))?;

            // the small offset keeps ln(0) from giving an infinite loss
            let p = outputs[target] + 1e-7;
            loss += -p.ln();
            grads[target] = -1.0 / p;
        }

        Ok((loss, grads.into_boxed_slice()))
    }

    /// Picks the most likely option of each output column.
    fn decode_outputs(&self, outputs: &[f32]) -> anyhow::Result<Vec<(usize, DataStorage)>> {
        self.column_slices()?
            .into_iter()
            .map(|(column, options, range)| {
                let best = outputs[range]
                    .iter()
                    .enumerate()
                    .max_by(|a, b| a.1.total_cmp(b.1))
                    .map(|(i, _)| i)
                    .with_context(|| format!("output column {} has no options", column.name()))?;
                Ok((column.index(), DataStorage::Enum(options[best].clone())))
            })
            .collect()
    }
}
