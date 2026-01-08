#[cfg(test)]
mod tests {
    use burn::backend::NdArray;
    use burn::tensor::activation::softmax;
    use burn::tensor::{Int, TensorData, Tolerance};
    use burn::{Tensor, backend::ndarray::NdArrayDevice};
    use log::info;
    use rstest::{fixture, rstest};

    #[fixture]
    #[once]
    fn init_logger() -> () {
        _ = env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .try_init();
    }

    #[rstest]
    fn test_self_attention(#[expect(unused_variables)] init_logger: &()) {
        type Backend = NdArray;
        let device = &NdArrayDevice::Cpu;
        let inputs: Tensor<Backend, 2> = Tensor::from_floats(
            [
                [0.43, 0.15, 0.89], // Your (x^1)
                [0.55, 0.87, 0.66], // journey (x^2)
                [0.57, 0.85, 0.64], // starts (x^3)
                [0.22, 0.58, 0.33], // with (x^4)
                [0.77, 0.25, 0.10], // one (x^5)
                [0.05, 0.80, 0.55], // step (x^6)
            ],
            device,
        );

        info!(input_shape:? = inputs.shape(); "defined input");
        let query = inputs
            .clone()
            .select(0, Tensor::<Backend, 1, Int>::from_data([1], device));

        let mut attn_scores_2: Tensor<Backend, 1> =
            burn::Tensor::empty([inputs.shape().dims[0]], device);

        for (i, x_i) in inputs.iter_dim(0).enumerate() {
            let x_i_squeezed = x_i.clone().squeeze::<1>();
            let score = x_i_squeezed.dot(query.clone().squeeze::<1>());

            attn_scores_2.inplace(|t| t.select_assign(0, Tensor::from_ints([i], device), score));
        }

        info!(attn_scores_2:?; "calculated attention scores");

        attn_scores_2.to_data().assert_approx_eq(
            &TensorData::new(
                vec![0.9544, 1.4950, 1.4754, 0.8434, 0.7070, 1.0865],
                vec![6],
            ),
            Tolerance::<f32>::balanced(),
        );

        let attn_weights_2_tmp = attn_scores_2.clone() / attn_scores_2.clone().sum();

        attn_weights_2_tmp.to_data().assert_approx_eq(
            &TensorData::new(
                vec![0.1455, 0.2278, 0.2249, 0.1285, 0.1077, 0.1656],
                vec![6],
            ),
            Tolerance::<f32>::balanced(),
        );

        attn_weights_2_tmp.sum().to_data().assert_approx_eq(
            &TensorData::new(vec![1.00], vec![1]),
            Tolerance::<f32>::balanced(),
        );

        let attn_weights_2 = softmax(attn_scores_2, 0);

        attn_weights_2.to_data().assert_approx_eq(
            &TensorData::new(
                vec![0.1385, 0.2379, 0.2333, 0.1240, 0.1082, 0.1581],
                vec![6],
            ),
            Tolerance::<f32>::balanced(),
        );

        attn_weights_2.sum().to_data().assert_approx_eq(
            &TensorData::new(vec![1.00], vec![1]),
            Tolerance::<f32>::balanced(),
        );
    }
}
