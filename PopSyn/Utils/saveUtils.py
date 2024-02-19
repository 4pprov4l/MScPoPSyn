from Utils import tuUtils
import numpy as np
import tensorflow as tf
import keras


def saveGenerator(generator_path, _WGAN):
    generator = _WGAN.generator
    generator_path = generator_path
    generator.save(generator_path)

def sampleGen(loaded_generator, n_samples, one_hot_n_col ,wgan_latent_dim, numerical_col_n, categories_cum, samp_df, col_names, pre_one_hot_df):
    z_sample = np.random.normal(0., 1.0, size=(n_samples, wgan_latent_dim))
    prediction = loaded_generator.predict(z_sample).transpose()

    samples = np.zeros((one_hot_n_col, n_samples))
    samples[:numerical_col_n,:]=prediction[:numerical_col_n,:]
    for idx in range(len(categories_cum)-1):
        idx_i = numerical_col_n+categories_cum[idx] # Initial index
        idx_f = numerical_col_n+categories_cum[idx+1] # Final index
        mask = np.argmax(prediction[idx_i:idx_f, :], axis=0) + idx_i
        samples[mask, np.arange(len(mask))] = 1

    wgan_n  = tuUtils.samples_to_df(samples, print_duplicates=False, col_names=col_names, original_df=samp_df, pre_one_hot_df=pre_one_hot_df)
    return wgan_n

def loadModel(generator_path):
    return tf.keras.models.load_model(generator_path)