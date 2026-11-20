use getset::Getters;

use crate::layers::Layer;

/// To avoid different elements having different spreads (diff between lowest and highest value)
/// from having an outsized effect on the model, we scale them all with scalars to have the same
/// overall impact on the model.  That is the goal of this structure
#[derive(Getters, Clone, Debug)]
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
