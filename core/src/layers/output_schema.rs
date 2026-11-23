use anyhow::Context;
use getset::Getters;
use ndarray::{Array2, ArrayView1, ArrayView2, Axis, Zip, s};
use serde::{Deserialize, Serialize};

use crate::{
    data::{DataConverter, DataSheet, DataStorage},
    layers::{Layer, input_schema::{SchemaColumn, encode_columns}}
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

    /// Applies a softmax to the scores of each output column of every row.
    fn transform(&self, scores: ArrayView2<f32>) -> anyhow::Result<Array2<f32>> {
        let mut probabilities = scores.to_owned();
        for (_, _, range) in self.column_slices()? {
            anyhow::ensure!(
                range.end <= scores.ncols(),
                "OutputSchema expected at least {} scores per row but got {}", range.end, scores.ncols()
            );

            // exponentiate then divide by the column's total, row by row
            let mut column_scores = probabilities.slice_mut(s![.., range]);
            for mut row in column_scores.rows_mut() {
                let max = row.fold(f32::NEG_INFINITY, |max, &score| max.max(score));
                row.mapv_inplace(|score| (score - max).exp());
                let sum = row.sum();
                row.mapv_inplace(|exp| exp / sum);
            }
        }

        Ok(probabilities)
    }

    /// Passes the gradients back through each column's softmax.
    fn back_propagate(
        &mut self,
        _inputs: ArrayView2<f32>,
        outputs: ArrayView2<f32>,
        output_gradient: ArrayView2<f32>
    ) -> anyhow::Result<Array2<f32>> {
        anyhow::ensure!(
            outputs.dim() == output_gradient.dim(),
            "OutputSchema got {:?} outputs but {:?} gradients", outputs.dim(), output_gradient.dim()
        );

        let mut input_grads = Array2::zeros(outputs.raw_dim());
        for (_, _, range) in self.column_slices()? {
            let column_outputs = outputs.slice(s![.., range.clone()]);
            let column_grads = output_gradient.slice(s![.., range.clone()]);

            // Σ g·y appears in every input's gradient, so compute it once per row
            let dots = (&column_grads * &column_outputs)
                .sum_axis(Axis(1))
                .insert_axis(Axis(1));

            // dL/dz_i = y_i · (g_i − Σ g·y)
            input_grads
                .slice_mut(s![.., range])
                .assign(&(&column_outputs * &(&column_grads - &dots)));
        }

        Ok(input_grads)
    }

    /// One-hot encodes the expected option of every output column of each row.
    fn encode_targets(&self, rows: &[&[DataStorage]]) -> anyhow::Result<Array2<f32>> {
        let Some(columns) = &self.columns else { anyhow::bail!("OutputSchema layer not prepared") };
        encode_columns(columns, rows, "output")
    }

    /// Cross-entropy of each output column's probabilities against the row's option, summed over
    /// the columns and rows, with its gradient.
    fn output_loss(
        &self,
        outputs: ArrayView2<f32>,
        targets: ArrayView2<f32>
    ) -> anyhow::Result<(f32, Array2<f32>)> {
        anyhow::ensure!(
            outputs.dim() == targets.dim(),
            "OutputSchema got {:?} outputs but {:?} targets", outputs.dim(), targets.dim()
        );

        // only the expected option of each column scores, the one-hot target picks it out
        let mut loss = 0.0;
        let mut grads = Array2::zeros(outputs.raw_dim());
        Zip::from(&mut grads)
            .and(&outputs)
            .and(&targets)
            .for_each(|grad, &output, &target| {
                if target > 0.0 {
                    // the small offset keeps ln(0) from giving an infinite loss
                    let p = output + 1e-7;
                    loss += -p.ln();
                    *grad = -1.0 / p;
                }
            });

        Ok((loss, grads))
    }

    /// Counts the rows where the most likely option of every output column is the expected one.
    fn count_correct(
        &self,
        outputs: ArrayView2<f32>,
        targets: ArrayView2<f32>
    ) -> anyhow::Result<usize> {
        anyhow::ensure!(
            outputs.dim() == targets.dim(),
            "OutputSchema got {:?} outputs but {:?} targets", outputs.dim(), targets.dim()
        );

        let slices = self.column_slices()?;
        let mut correct = 0;
        for (output_row, target_row) in outputs.rows().into_iter().zip(targets.rows()) {
            let all_match = slices
                .iter()
                .all(|(_, _, range)| {
                    best_option(output_row.slice(s![range.clone()])) == best_option(target_row.slice(s![range.clone()]))
                });
            if all_match { correct += 1; }
        }

        Ok(correct)
    }

    /// Picks the most likely option of each output column of every row.
    fn decode_outputs(&self, outputs: ArrayView2<f32>) -> anyhow::Result<Vec<Vec<(usize, DataStorage)>>> {
        let slices = self.column_slices()?;
        outputs
            .rows()
            .into_iter()
            .map(|row| {
                slices
                    .iter()
                    .map(|(column, options, range)| {
                        anyhow::ensure!(
                            range.end <= row.len(),
                            "OutputSchema expected at least {} outputs per row but got {}", range.end, row.len()
                        );
                        let best = best_option(row.slice(s![range.clone()]))
                            .with_context(|| format!("output column {} has no options", column.name()))?;
                        Ok((column.index(), DataStorage::Enum(options[best].clone())))
                    })
                    .collect()
            })
            .collect()
    }
}

/// Index of the highest value in one column's scores, `None` if the column has no options.
fn best_option(scores: ArrayView1<f32>) -> Option<usize> {
    scores
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
}
