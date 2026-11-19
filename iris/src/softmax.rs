
#[derive(Copy, Clone, Debug)]
pub struct SoftMax<const DEPTH: usize>;

impl <const DEPTH: usize> SoftMax<DEPTH> {

    /// Turn some random result scores into a probably scaled from [0, 1]
    /// with the total sum of the output being 1.0.
    pub fn transform(&self, scores: &[f32; DEPTH]) -> [f32; DEPTH] {
        let mut sum = 0.0;
        let mut exps = [0.0; DEPTH];
        let max = scores.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

        for c in 0..DEPTH {
            exps[c] = (scores[c] - max).exp();
            sum += exps[c];
        }

        for c in 0..DEPTH {
            exps[c] /= sum;
        }

        return exps;
    }
}
