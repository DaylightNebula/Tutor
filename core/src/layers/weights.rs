use getset::{CopyGetters, Getters};
use ndarray::{Array1, Array2, ArrayView2, Axis};
use ndarray_rand::{RandomExt, rand_distr::Uniform};
use serde::{Deserialize, Serialize};

use crate::{data::DataSheet, layers::Layer};

/// Weights (fully connected / dense / linear layer)
///
/// # Purpose
///
/// Holds the raw, learnable numbers of the model.  Every output is a weighted sum of every
/// input plus a bias, so this is the layer that actually "learns" during training.  Its outputs
/// are raw scores rather than probabilities; follow it with a layer like
/// [`SoftMax`](crate::layers::softmax::SoftMax) to turn them into probabilities, or with
/// [`ZeroNegative`](crate::layers::zero_neg::ZeroNegative) to build a hidden layer.
///
/// # How it works
///
/// With an input vector x of length `width`, a weight matrix W of size `depth` × `width`, and a
/// bias vector b of length `depth`, the output is:
///
/// ```text
/// y = W·x + b        i.e.   y_c = b_c + Σ_{j=1..width} W_cj · x_j
/// ```
///
/// During back propagation, given g_c = ∂L/∂y_c, the chain rule gives:
///
/// ```text
/// ∂L/∂W_cj = g_c · x_j          (gradient for each weight)
/// ∂L/∂b_c  = g_c                (gradient for each bias)
/// ∂L/∂x_j  = Σ_{c=1..depth} W_cj · g_c   (gradient passed to the previous layer)
/// ```
///
/// Rows are processed as a batch, an input matrix X of size `rows` × `width` whose output is
/// Y = X·Wᵀ + b, and the gradients above become ∂L/∂W = Gᵀ·X, ∂L/∂b = the column sums of G, and
/// ∂L/∂X = G·W, where G holds every row's g.  These are summed over every row of a training
/// batch.  After each batch, with n rows in the batch, learning rate η, and weight decay λ,
/// gradient descent updates the parameters:
///
/// ```text
/// W_cj ← W_cj − η · ( (Σ ∂L/∂W_cj) / n + λ · W_cj )
/// b_c  ← b_c  − η · (Σ ∂L/∂b_c) / n
/// ```
///
/// Weights start random in the range ±sqrt(2 / width) (He initialization) and biases start at 0.
///
/// In plain algebra:
/// * Each output is like a line equation y = m·x + b, but with many inputs: multiply each input
///   by its own weight, add them all up, then add the bias.  Example with weights [0.5, −2] and
///   bias 1: inputs [4, 1] give 0.5·4 + (−2)·1 + 1 = 1.
/// * A large positive weight means "this input pushes this output up a lot", a negative weight
///   means "this input pushes it down", and a weight near 0 means "this input doesn't matter".
/// * Training asks "if I nudge this weight slightly, does the error go up or down, and by how
///   much?"  That answer is the gradient.  For a weight it is simply (how wrong the output was)
///   × (the input that weight multiplied), because the input is what the weight was scaled by.
/// * After each batch of rows, every weight takes a small step (sized by the learning rate) in
///   the direction that lowers the error, using the average gradient over the batch's rows.
/// * Weight decay also shrinks every weight a little each step, which keeps weights from growing
///   huge and helps the model generalize instead of memorizing the training data.
///
/// # Example uses
/// * Scoring each iris species from four flower measurements (width 4, depth 3).
/// * Hidden layers in a deeper network when stacked with an activation layer between them.
/// * Linear regression, predicting a number such as a house price from its features (depth 1).
///
/// Fields:
/// * contents: The learned weights and biases.  `None` until `prepare_train` sizes
///   it from the width of the layer before it.
/// * training_info: Running gradient totals for the current batch.  Created on the first back
///   propagation and not serialized.
/// * depth: Number of outputs.
/// * weight_decay: Strength (λ) of the pull toward zero applied to weights each step.
#[derive(CopyGetters, Getters, Clone, Debug, Serialize, Deserialize)]
pub struct Weights {
    #[getset(get = "pub")]
    contents: Option<WeightContents>,
    #[serde(skip, default)]
    #[getset(get = "pub")]
    training_info: Option<WeightsTrainingInfo>,
    #[getset(get_copy = "pub")]
    depth: usize,
    #[getset(get_copy = "pub")]
    weight_decay: f32
}

/// The learned parameters of a [`Weights`] layer, sized once its input width is known.
///
/// Fields:
/// * weights: The `width` × `depth` matrix; `weights[[j, c]]` scales input j for output c.  It is
///   stored as the transpose of W so a batch of inputs multiplies it directly.
/// * biases: One constant per output, added after the weighted sum.
#[derive(Getters, Clone, Debug, Serialize, Deserialize)]
pub struct WeightContents {
    #[getset(get = "pub")]
    weights: Array2<f32>,
    #[getset(get = "pub")]
    biases: Array1<f32>
}

impl WeightContents {
    /// Number of inputs, taken from the output size of the layer before this one.
    pub fn width(&self) -> usize {
        self.weights.nrows()
    }
}

/// Gradient totals a [`Weights`] layer collects over one batch of training.
///
/// Fields:
/// * gradient_weights: Running total of weight gradients, the same `width` × `depth` shape as the
///   weights.
/// * gradient_biases: Running total of bias gradients, one per output.
#[derive(Clone, Debug)]
pub struct WeightsTrainingInfo {
    gradient_weights: Array2<f32>,
    gradient_biases: Array1<f32>
}

impl Weights {
    /// Create a new weights instance of the given size.
    pub fn random(num_outputs: usize, weight_decay: f32) -> Self {
        Self {
            contents: None,
            training_info: None,
            depth: num_outputs, weight_decay
        }
    }
}

#[typetag::serde]
impl Layer for Weights {

    fn output_size(&self, _input_count: usize) -> usize { self.depth }

    /// Sizes the weights to `input_count` inputs with fresh random values and clears any gradients
    /// left over from a previous training run.
    fn prepare_train(
        &mut self,
        _sample_data: &DataSheet,
        input_count: usize
    ) -> anyhow::Result<()> {
        // he initialization scaled to the number of inputs
        let scale = (2.0 / input_count as f32).sqrt();
        let weights = Array2::random((input_count, self.depth), Uniform::new(-scale, scale));

        self.contents = Some(WeightContents {
            weights,
            biases: Array1::zeros(self.depth)
        });
        self.training_info = None;
        Ok(())
    }

    /// Use the model to take a batch of <width> inputs, score, then return <depth> outputs per row.
    ///
    /// The output numbers from this will be kind of random, not a probability, use
    /// a function like a softmax to turn it into a probability of one output being the
    /// most likely.
    ///
    /// # Errors
    /// * The layer has not been prepared.
    /// * The inputs do not have `width` columns.
    fn transform(&self, inputs: ArrayView2<f32>) -> anyhow::Result<Array2<f32>> {
        let content = prepared_contents(&self.contents, inputs)?;

        // weighted sum of every input plus the bias for each output, for every row at once
        Ok(inputs.dot(&content.weights) + &content.biases)
    }

    /// Adds the batch's weight and bias gradients to the running totals and gives dLoss/dInput.
    ///
    /// # Errors
    /// * The layer has not been prepared.
    /// * The inputs do not have `width` columns or the gradients do not have `depth` columns.
    fn back_propagate(
        &mut self,
        inputs: ArrayView2<f32>,
        _outputs: ArrayView2<f32>,
        output_grads: ArrayView2<f32>
    ) -> anyhow::Result<Array2<f32>> {
        let content = prepared_contents(&self.contents, inputs)?;
        anyhow::ensure!(
            output_grads.dim() == (inputs.nrows(), self.depth),
            "Weights layer expected {} × {} output gradients but got {:?}", inputs.nrows(), self.depth, output_grads.dim()
        );

        // gradient totals start at zero on the first batch of training
        let training_info = self.training_info.get_or_insert_with(|| WeightsTrainingInfo {
            gradient_weights: Array2::zeros(content.weights.raw_dim()),
            gradient_biases: Array1::zeros(content.biases.raw_dim())
        });

        // add this batch's parameter gradients and pass dLoss/dInput back
        training_info.gradient_weights += &inputs
            .t()
            .dot(&output_grads);
        training_info.gradient_biases += &output_grads.sum_axis(Axis(0));
        Ok(output_grads.dot(&content.weights.t()))
    }

    /// Steps the weights against the averaged gradients, then clears them for the next batch.
    fn perform_epoch_step(&mut self, learning_rate: f32, n: f32) -> anyhow::Result<()> {
        let Some(content) = self.contents.as_mut() else { anyhow::bail!("Weights layer not prepared before training") };
        let Some(training_info) = self.training_info.as_mut() else { anyhow::bail!("Training info not set, likely a step called before back propagate") };

        // step against the average gradient, with weight decay pulling every weight toward zero
        let weight_decay = self.weight_decay;
        content.weights.zip_mut_with(&training_info.gradient_weights, |weight, gradient| {
            *weight -= learning_rate * (gradient / n + weight_decay * *weight);
        });
        content.biases.zip_mut_with(&training_info.gradient_biases, |bias, gradient| {
            *bias -= learning_rate * gradient / n;
        });

        training_info.gradient_weights.fill(0.0);
        training_info.gradient_biases.fill(0.0);
        Ok(())
    }
}

/// The prepared weights, checked against the width of a batch of `inputs` so the matrix products
/// cannot panic on mismatched shapes.  Takes the `contents` field rather than the layer so the
/// layer's training info can still be borrowed mutably.
fn prepared_contents<'a>(
    contents: &'a Option<WeightContents>,
    inputs: ArrayView2<f32>
) -> anyhow::Result<&'a WeightContents> {
    let Some(content) = contents.as_ref() else { anyhow::bail!("Weights layer not prepared before training") };
    anyhow::ensure!(
        inputs.ncols() == content.width(),
        "Weights layer expected {} inputs per row but got {}", content.width(), inputs.ncols()
    );
    Ok(content)
}
