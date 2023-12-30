use itertools::Itertools;

#[derive(Debug, Clone)]
pub struct Sampler {
    pub top_p: f32,
    pub temperature: f32,
    pub presence_penalty: f32,
    pub frequency_penalty: f32,
    pub penalty_decay: f32,
}

impl Default for Sampler {
    fn default() -> Self {
        Self {
            top_p: 1.0,
            temperature: 1.0,
            presence_penalty: 0.0,
            frequency_penalty: 0.0,
            penalty_decay: 1.0,
        }
    }
}

impl Sampler {
    pub fn sample(&self, probs: Vec<f32>) -> u16 {
        let sorted = probs
            .into_iter()
            .enumerate()
            .sorted_unstable_by(|(_, x), (_, y)| x.total_cmp(y).reverse())
            .scan((0, 0.0, 0.0), |(_, cum, _), (id, x)| {
                if *cum > self.top_p {
                    None
                } else {
                    *cum += x;
                    Some((id, *cum, x))
                }
            })
            .map(|(id, _, x)| (id, x.powf(1.0 / self.temperature)))
            .collect_vec();

        let sum: f32 = sorted.iter().map(|(_, x)| x).sum();
        let sorted = sorted
            .into_iter()
            .map(|(id, x)| (id, x / sum))
            .scan((0, 0.0), |(_, cum), (id, x)| {
                *cum += x;
                Some((id, *cum))
            })
            .collect_vec();

        let rand = fastrand::f32();
        let token = sorted
            .into_iter()
            .find_or_first(|&(_, cum)| rand <= cum)
            .map(|(id, _)| id)
            .unwrap_or_default();
        token as u16
    }
}
