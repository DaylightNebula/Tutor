use getset::{CopyGetters, Getters};
use rand::Rng;
use serde::{Deserialize, Serialize};

use crate::layers::Layer;

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
/// These are summed over every training row.  At the end of each epoch, with n rows, learning
/// rate η, and weight decay λ, gradient descent updates the parameters:
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
/// * Each epoch, every weight takes a small step (sized by the learning rate) in the direction
///   that lowers the error, using the average gradient over all rows.
/// * Weight decay also shrinks every weight a little each step, which keeps weights from growing
///   huge and helps the model generalize instead of memorizing the training data.
///
/// # Example uses
/// * Scoring each iris species from four flower measurements (width 4, depth 3).
/// * Hidden layers in a deeper network when stacked with an activation layer between them.
/// * Linear regression, predicting a number such as a house price from its features (depth 1).
///
/// Fields:
/// * contents: The learned weights, biases, and input width.  `None` until `prepare_train` sizes
///   it from the width of the layer before it.
/// * training_info: Running gradient totals for the current epoch.  Created on the first back
///   propagation and not serialized.
/// * depth: Number of outputs.
/// * weight_decay: Strength (λ) of the pull toward zero applied to weights each epoch step.
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
/// * weights: The `depth` × `width` matrix; `weights[c][j]` scales input j for output c.
/// * biases: One constant per output, added after the weighted sum.
/// * width: Number of inputs, taken from the output size of the layer before this one.
#[derive(Getters, CopyGetters, Clone, Debug, Serialize, Deserialize)]
pub struct WeightContents {
    #[getset(get = "pub")]
    weights: Vec<Vec<f32>>,
    #[getset(get = "pub")]
    biases: Box<[f32]>,
    #[getset(get_copy = "pub")]
    width: usize
}

/// Gradient totals a [`Weights`] layer collects over one epoch of training.
///
/// Fields:
/// * gradient_weights: Running total of weight gradients, the same `depth` × `width` shape as the
///   weights.
/// * gradient_biases: Running total of bias gradients, one per output.
#[derive(Clone, Debug)]
pub struct WeightsTrainingInfo {
    gradient_weights: Vec<Vec<f32>>,
    gradient_biases: Box<[f32]>
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
        _sample_data: &[(Box<[f32]>, usize)],
        input_count: usize
    ) -> anyhow::Result<()> {
        // he initialization scaled to the number of inputs
        let mut rng = rand::thread_rng();
        let scale = (2.0 / input_count as f32).sqrt();
        let weights = (0 .. self.depth)
            .map(|_| {
                (0 .. input_count)
                    .map(|_| rng.gen_range(-1.0 .. 1.0) * scale)
                    .collect()
            })
            .collect();

        self.contents = Some(WeightContents {
            weights,
            biases: vec![0.0; self.depth].into_boxed_slice(),
            width: input_count
        });
        self.training_info = None;
        Ok(())
    }

    /// Use the model to take <width> inputs, score, then return the <depth> outputs.
    ///
    /// The output numbers from this will be kind of random, not a probability, use
    /// a function like a softmax to turn it into a probability of one output being the
    /// most likely.
    fn transform(&self, input: &[f32]) -> anyhow::Result<Box<[f32]>> {
        let Some(content) = self.contents.as_ref() else { anyhow::bail!("Weights layer not prepared before training") };

        // weighted sum of every input plus the bias for each output
        let mut s = vec![0.0; self.depth];
        for c in 0 .. self.depth {
            s[c] = content.biases[c];
            for j in 0 .. content.width {
                s[c] += content.weights[c][j] * input[j];
            }
        }

        return Ok(s.into_boxed_slice());
    }

    fn back_propagate(
        &mut self,
        inputs: &[f32],
        _outputs: &[f32],
        output_grads: &[f32]
    ) -> anyhow::Result<Box<[f32]>> {
        let Some(content) = self.contents.as_ref() else { anyhow::bail!("Weights layer not prepared before training") };

        // gradient totals start at zero on the first row of training
        let depth = self.depth;
        let training_info = self.training_info.get_or_insert_with(|| WeightsTrainingInfo {
            gradient_weights: vec![vec![0.0; content.width]; depth],
            gradient_biases: vec![0.0; depth].into_boxed_slice()
        });

        // add this row's parameter gradients and pass dLoss/dInput back
        let mut input_grads = vec![0.0; content.width];
        for c in 0 .. self.depth {
            for j in 0 .. content.width {
                training_info.gradient_weights[c][j] += output_grads[c] * inputs[j];
                input_grads[j] += content.weights[c][j] * output_grads[c];
            }
            training_info.gradient_biases[c] += output_grads[c];
        }

        Ok(input_grads.into_boxed_slice())
    }

    /// Steps the weights against the averaged gradients, then clears them for the next epoch.
    fn perform_epoch_step(&mut self, learning_rate: f32, n: f32) -> anyhow::Result<()> {
        let Some(content) = self.contents.as_mut() else { anyhow::bail!("Weights layer not prepared before training") };
        let Some(training_info) = self.training_info.as_mut() else { anyhow::bail!("Training info not set, likely epoch end step called before back propagate") };

        for c in 0 .. self.depth {
            for j in 0 .. content.width {
                content.weights[c][j] -= learning_rate * (training_info.gradient_weights[c][j] / n + self.weight_decay * content.weights[c][j]);
                training_info.gradient_weights[c][j] = 0.0;
            }
            content.biases[c] -= learning_rate * training_info.gradient_biases[c] / n;
            training_info.gradient_biases[c] = 0.0;
        }

        Ok(())
    }
}
