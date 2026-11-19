use crate::layers::Layer;


#[derive(Copy, Clone, Debug)]
pub struct SoftMax {
    num_inputs: usize
}

impl SoftMax {
    pub fn new(num_inputs: usize) -> Self {
        Self { num_inputs }
    }
}

impl Layer for SoftMax {
    fn input_size(&self) -> usize {
        self.num_inputs
    }

    fn output_size(&self) -> usize {
        self.num_inputs
    }

    /// Turn some random result scores into a probably scaled from [0, 1]
    /// with the total sum of the output being 1.0.
    fn transform(&self, scores: &[f32]) -> Box<[f32]> {
        let mut sum = 0.0;
        let mut exps = vec![0.0; self.num_inputs];
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        for c in 0 .. self.num_inputs {
            exps[c] = (scores[c] - max).exp();
            sum += exps[c];
        }

        for c in 0 .. self.num_inputs {
            exps[c] /= sum;
        }

        return exps.into_boxed_slice();
    }
}
