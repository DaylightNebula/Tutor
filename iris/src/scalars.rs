use getset::CopyGetters;

/// To avoid different elements having different spreads (diff between lowest and highest value)
/// from having an outsized effect on the model, we scale them all with scalars to have the same
/// overall impact on the model.  That is the goal of this structure
#[derive(CopyGetters, Clone, Copy, Debug)]
pub struct Scalars<const WIDTH: usize> {
    #[getset(get_copy = "pub")]
    mean: [f32; WIDTH],
    #[getset(get_copy = "pub")]
    standard_deviation: [f32; WIDTH],
}

impl <const WIDTH: usize> Scalars<WIDTH> {

    /// Find the best mean and standard deviation values from a set of rows to
    /// make ensure each column will be translated to a normalized roughly equal
    /// range.
    pub fn build_from_fit(data: &[[f32; WIDTH]]) -> Self {
        let n = data.len() as f32;

        let mut mean = [0.0; WIDTH];
        for f in data {
            for j in 0 .. WIDTH {
                mean[j] += f[j] / n;
            }
        }

        let mut std = [0.0; WIDTH];
        for f in data {
            for j in 0 .. WIDTH {
                std[j] += (f[j] - mean[j]).powi(2) / n;
            }
        }

        for j in 0 .. WIDTH {
            std[j] = std[j].sqrt();
        }

        return Scalars { mean, standard_deviation: std };
    }

    /// Take some unscaled numbers and scale them using the mean and standard
    /// deviation in this struct into the [-1, 1] range, assuming that this Scalars
    /// instance was given data containing roughly the lowest and highest values
    /// when it was built.
    pub fn transform(&self, unscaled: &[f32; WIDTH]) -> [f32; WIDTH] {
        let mut out = [0.0; WIDTH];
        for j in 0 .. WIDTH {
            out[j] = (unscaled[j] - self.mean[j]) / self.standard_deviation[j];
        }
        return out;
    }
}
