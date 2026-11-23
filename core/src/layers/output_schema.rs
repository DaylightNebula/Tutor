use anyhow::Context;
use getset::Getters;
use ndarray::{Array2, ArrayView1, ArrayView2, s};
use serde::{Deserialize, Serialize};

use crate::{
    data::{DataConverter, DataSheet, DataStorage},
    layers::{Layer, input_schema::{SchemaColumn, encode_columns}},
    model::ColumnScore
};

/// Turns the raw scores of the layer before it into predictions for every output column, and
/// reads them back into [`DataSheet`] values.  It must be the last layer of a model.
///
/// Enum columns get one score per option, turned into probabilities with a softmax, so this
/// layer replaces a separate softmax layer.  Decimal and integer columns get a single score that
/// is used as the column's normalized value directly, so a model can classify and regress at the
/// same time.
///
/// Fields:
/// * columns: The output columns and their converters, in the order their scores appear.  `None`
///   until `prepare_train` fits them to the sample data.
#[derive(Getters, Clone, Debug, Default, Serialize, Deserialize)]
pub struct OutputSchema {
    #[getset(get = "pub")]
    columns: Option<Vec<SchemaColumn>>
}

/// How an output column's scores are predicted and scored.
///
/// Variants:
/// * Enum: One score per option, turned into probabilities by a softmax and scored with
///   cross-entropy.
/// * Numeric: A single score that is the column's normalized value, scored with squared error.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColumnKind {
    Enum,
    Numeric
}

/// An output column, how it is predicted, and the range of this layer's scores that belong to it.
type ColumnSlice<'a> = (&'a SchemaColumn, ColumnKind, std::ops::Range<usize>);

impl OutputSchema {
    /// Creates an output schema layer that is fitted to its columns when the model is built.
    pub fn new() -> Self { Self::default() }

    /// The prepared columns, each paired with its kind and its slice of this layer's scores.
    fn column_slices(&self) -> anyhow::Result<Vec<ColumnSlice<'_>>> {
        let Some(columns) = &self.columns else { anyhow::bail!("OutputSchema layer not prepared") };

        let mut offset = 0;
        let mut slices = Vec::with_capacity(columns.len());
        for column in columns {
            let kind = match column.converter() {
                DataConverter::Enum { .. } => ColumnKind::Enum,
                DataConverter::Decimal { .. } | DataConverter::Integer { .. } => ColumnKind::Numeric
            };
            let width = column.converter().num_weights();
            slices.push((column, kind, offset .. offset + width));
            offset += width;
        }

        Ok(slices)
    }

    /// The column slices, checked against the number of scores per row in a batch so slicing
    /// cannot panic.
    fn checked_slices(&self, scores_per_row: usize) -> anyhow::Result<Vec<ColumnSlice<'_>>> {
        let slices = self.column_slices()?;
        let width = slices
            .last()
            .map_or(0, |(_, _, range)| range.end);
        anyhow::ensure!(
            width == scores_per_row,
            "OutputSchema expected {width} scores per row but got {scores_per_row}"
        );
        Ok(slices)
    }
}

#[typetag::serde]
impl Layer for OutputSchema {

    /// One score per option of every enum output column plus one per numeric column.  Before the
    /// schema is prepared the width is not known yet, so the previous layer's width is passed
    /// through.
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

        // the layer before must give one score per enum option and per numeric column
        let width = self.output_size(previous_layer_num_outputs);
        anyhow::ensure!(
            width == previous_layer_num_outputs,
            "the output columns need {width} scores in total but the layer before OutputSchema gives {previous_layer_num_outputs} scores"
        );

        Ok(())
    }

    /// Applies a softmax to the scores of each enum output column of every row.  Numeric columns
    /// keep their score as their normalized value.
    fn transform(&self, scores: ArrayView2<f32>) -> anyhow::Result<Array2<f32>> {
        let slices = self.checked_slices(scores.ncols())?;
        let mut outputs = scores.to_owned();
        for (_, kind, range) in slices {
            if kind != ColumnKind::Enum { continue; }

            // exponentiate then divide by the column's total, row by row
            let mut column_scores = outputs.slice_mut(s![.., range]);
            for mut row in column_scores.rows_mut() {
                let max = row.fold(f32::NEG_INFINITY, |max, &score| max.max(score));
                row.mapv_inplace(|score| (score - max).exp());
                let sum = row.sum();
                row.mapv_inplace(|exp| exp / sum);
            }
        }

        Ok(outputs)
    }

    /// Passes the gradients back unchanged.
    ///
    /// `output_loss` already gives dLoss/dScore, the gradient with respect to this layer's
    /// inputs, since the softmax and cross-entropy of an enum column simplify together to
    /// `probability − target`.  This only holds when the gradients come from this layer's own
    /// `output_loss`.
    fn back_propagate(
        &mut self,
        inputs: ArrayView2<f32>,
        _outputs: ArrayView2<f32>,
        output_gradient: ArrayView2<f32>
    ) -> anyhow::Result<Array2<f32>> {
        anyhow::ensure!(
            inputs.dim() == output_gradient.dim(),
            "OutputSchema got {:?} inputs but {:?} gradients", inputs.dim(), output_gradient.dim()
        );
        Ok(output_gradient.to_owned())
    }

    /// Encodes the expected value of every output column of each row: one-hot for enum columns
    /// and normalized for numeric ones.
    fn encode_targets(&self, rows: &[&[DataStorage]]) -> anyhow::Result<Array2<f32>> {
        let Some(columns) = &self.columns else { anyhow::bail!("OutputSchema layer not prepared") };
        encode_columns(columns, rows, "output")
    }

    /// Cross-entropy of each enum column's probabilities against the row's option, plus half the
    /// squared error of each numeric column's normalized value, summed over the columns and rows.
    ///
    /// The gradient given is with respect to the scores this layer received, not its outputs,
    /// and is `output − target` for both kinds of column, see `back_propagate`.
    fn output_loss(
        &self,
        outputs: ArrayView2<f32>,
        targets: ArrayView2<f32>
    ) -> anyhow::Result<(f32, Array2<f32>)> {
        anyhow::ensure!(
            outputs.dim() == targets.dim(),
            "OutputSchema got {:?} outputs but {:?} targets", outputs.dim(), targets.dim()
        );
        let slices = self.checked_slices(outputs.ncols())?;

        // add up each column's loss in the way that suits its kind
        let mut loss = 0.0;
        for (_, kind, range) in slices {
            let column_outputs = outputs.slice(s![.., range.clone()]);
            let column_targets = targets.slice(s![.., range]);
            loss += match kind {
                // the small offset keeps ln(0) from giving an infinite loss
                ColumnKind::Enum => -(&column_targets * &column_outputs.mapv(|p| (p + 1e-7).ln())).sum(),
                ColumnKind::Numeric => 0.5 * (&column_outputs - &column_targets)
                    .mapv(|difference| difference * difference)
                    .sum()
            };
        }

        Ok((loss, &outputs - &targets))
    }

    /// Scores each output column over every row: the accuracy of enum columns, and the mean
    /// absolute error in the column's units of numeric ones.
    fn evaluate(
        &self,
        outputs: ArrayView2<f32>,
        targets: ArrayView2<f32>
    ) -> anyhow::Result<Vec<ColumnScore>> {
        let rows = outputs.nrows() as f32;

        self.checked_slices(outputs.ncols())?
            .into_iter()
            .map(|(column, kind, range)| {
                let column_outputs = outputs.slice(s![.., range.clone()]);
                let column_targets = targets.slice(s![.., range]);
                match kind {
                    ColumnKind::Enum => {
                        // rows whose most likely option is the expected one
                        let correct = column_outputs
                            .rows()
                            .into_iter()
                            .zip(column_targets.rows())
                            .filter(|(output, target)| best_option(*output) == best_option(*target))
                            .count();
                        Ok(ColumnScore::Accuracy {
                            column: column.name().clone(),
                            accuracy: correct as f32 / rows
                        })
                    },
                    ColumnKind::Numeric => {
                        // distance between prediction and target, scaled back to the column's units
                        let mut total_error = 0.0;
                        for (output, target) in column_outputs.iter().zip(column_targets.iter()) {
                            let predicted = column.converter().denormalize(*output)?;
                            let expected = column.converter().denormalize(*target)?;
                            total_error += (predicted - expected).abs();
                        }
                        Ok(ColumnScore::MeanAbsoluteError {
                            column: column.name().clone(),
                            error: total_error / rows
                        })
                    }
                }
            })
            .collect()
    }

    /// Reads back the prediction of each output column of every row: the most likely option of
    /// enum columns, and the value in the column's units of numeric ones.
    fn decode_outputs(&self, outputs: ArrayView2<f32>) -> anyhow::Result<Vec<Vec<(usize, DataStorage)>>> {
        let slices = self.checked_slices(outputs.ncols())?;
        outputs
            .rows()
            .into_iter()
            .map(|row| {
                slices
                    .iter()
                    .map(|(column, _, range)| {
                        let encoded = row
                            .slice(s![range.clone()])
                            .to_vec();
                        let value = column.converter()
                            .decode(&encoded)
                            .with_context(|| format!("failed to decode output column {}", column.name()))?;
                        Ok((column.index(), value))
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
