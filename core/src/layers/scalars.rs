use getset::Getters;
use serde::{Deserialize, Serialize};

use crate::layers::Layer;

/// Scalars (standardization / z-score normalization layer)
///
/// # Purpose
///
/// Raw input columns often live on very different scales.  One column might range from 0 to 1
/// while another ranges from 1,000 to 50,000.  Without correction, the large-valued columns
/// would have an outsized effect on the model and make training slow and unstable.  This layer
/// rescales every column so each one is centered on 0 with a spread of about 1, giving every
/// input the same overall impact on the model.  It is normally the first layer in a model.
///
/// # How it works
///
/// The layer is fit once with [`Scalars::from_fit`], which learns a mean (μ) and standard
/// deviation (σ) for every column j over the N training rows:
///
/// ```text
/// μ_j = (1 / N) · Σ_{i=1..N} x_ij
/// σ_j = sqrt( (1 / N) · Σ_{i=1..N} (x_ij − μ_j)² )
/// ```
///
/// Each input is then transformed with the z-score formula:
///
/// ```text
/// y_j = (x_j − μ_j) / σ_j
/// ```
///
/// During back propagation the chain rule gives the gradient for the previous layer.  Since
/// `∂y_j / ∂x_j = 1 / σ_j`:
///
/// ```text
/// ∂L/∂x_j = (∂L/∂y_j) · (1 / σ_j)
/// ```
///
/// In plain algebra:
/// * The mean μ is the ordinary average: add up every value in the column and divide by how
///   many values there are.
/// * The standard deviation σ measures how spread out the column is.  For each value, find how
///   far it is from the average, square that distance (so negatives don't cancel positives),
///   average those squares, then take the square root to get back to the original units.
/// * The transform subtracts the average, so a typical value becomes 0, then divides by the
///   spread, so the result counts "how many spreads away from average" the value is.  For
///   example, with μ = 50 and σ = 10, the input 70 becomes (70 − 50) / 10 = 2, and the input
///   45 becomes (45 − 50) / 10 = −0.5.
/// * Going backwards, the layer only divided by σ, so any change at the output is also divided
///   by σ on its way back to the input.  There is nothing to learn here, so μ and σ never
///   change after fitting.
///
/// Most outputs land roughly in the [-3, 3] range, with about two thirds in [-1, 1] for
/// bell-curve shaped data.
///
/// # Example uses
/// * Normalizing flower measurements (petal and sepal lengths in cm) before classification.
/// * Putting features like age (0–100) and income (0–200,000) on equal footing.
/// * Normalizing sensor readings with different units (temperature, pressure, humidity).
///
/// Fields:
/// * mean: Average value of each input column, learned from the fit data.
/// * standard_deviation: Spread of each input column, learned from the fit data.  Each input is
///   divided by its column's value.
#[derive(Getters, Clone, Debug, Serialize, Deserialize)]
pub struct Scalars {
    #[getset(get_copy = "pub")]
    mean: Vec<f32>,
    #[getset(get_copy = "pub")]
    standard_deviation: Vec<f32>,
}

impl Scalars {

    /// Find the best mean and standard deviation values from a set of rows to
    /// make ensure each column will be translated to a normalized roughly equal
    /// range.
    pub fn from_fit(data: &Vec<Vec<f32>>) -> Self {
        let n = data.len() as f32;
        let w = data[0].len();

        let mut mean = vec![0.0; w];
        for f in data {
            for j in 0 .. w {
                mean[j] += f[j] / n;
            }
        }

        let mut std = vec![0.0; w];
        for f in data {
            for j in 0 .. w {
                std[j] += (f[j] - mean[j]).powi(2) / n;
            }
        }

        for j in 0 .. w {
            std[j] = std[j].sqrt();
        }

        return Scalars { mean, standard_deviation: std };
    }
}

#[typetag::serde]
impl Layer for Scalars {

    fn input_size(&self) -> usize {
        self.mean.len()
    }

    fn output_size(&self) -> usize {
        self.mean.len()
    }

    /// Take some unscaled numbers and scale them using the mean and standard
    /// deviation in this struct into the [-1, 1] range, assuming that this Scalars
    /// instance was given data containing roughly the lowest and highest values
    /// when it was built.
    fn transform(&self, unscaled: &[f32]) -> Box<[f32]> {
        let mut out = vec![0.0; unscaled.len()];
        for j in 0 .. unscaled.len() {
            out[j] = (unscaled[j] - self.mean[j]) / self.standard_deviation[j];
        }
        return out.into_boxed_slice();
    }

    /// Translate the gradients via the same equation as above to pass back the gradients
    /// for the previous layer.
    fn back_propagate(
        &mut self,
        _inputs: &[f32],
        _outputs: &[f32],
        output_gradient: &[f32]
    ) -> anyhow::Result<Box<[f32]>> {
        let input_grads = output_gradient
            .iter()
            .zip(&self.standard_deviation)
            .map(|(g, std)| g / std)
            .collect::<Vec<_>>();

        Ok(input_grads.into_boxed_slice())
    }
}
