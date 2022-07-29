import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from PendulumData import Pendulum
from GIN import GIN
from LayerNormalizer import LayerNormalizer


def generate_imputation_data_set(pendulum, num_seqs, seq_length, seed):
    obs, _, _, _ = pendulum.sample_data_set(num_seqs, seq_length, full_targets=False, seed=seed)
    obs = np.expand_dims(obs, -1)
    targets = obs.copy()

    rs = np.random.RandomState(seed=seed)
    obs_valid = rs.rand(num_seqs, seq_length, 1) < 0.5
    obs_valid[:, :20] = True
    print("Fraction of Valid Images:", np.count_nonzero(obs_valid) / np.prod(obs_valid.shape))
    obs[np.logical_not(np.squeeze(obs_valid))] = 0

    return obs.astype(np.float32), obs_valid, targets.astype(np.float32)


# Implement Encoder and Decoder hidden layers
class PendulumImageImputationGIN(GIN):

    def build_encoder_hidden(self):
        return [
            # 1: Conv Layer
            k.layers.Conv2D(12, kernel_size=5, padding="same"),
            LayerNormalizer(),
            k.layers.Activation(k.activations.relu),
            k.layers.MaxPool2D(2, strides=2),
            # 2: Conv Layer
            k.layers.Conv2D(12, kernel_size=3, padding="same", strides=2),
            LayerNormalizer(),
            k.layers.Activation(k.activations.relu),
            k.layers.MaxPool2D(2, strides=2),
            k.layers.Flatten(),
            # 3: Dense Layer
            k.layers.Dense(30, activation=k.activations.relu)]

    def build_decoder_hidden(self):
        return [
            k.layers.Dense(144, activation=k.activations.relu),
            k.layers.Lambda(lambda x: tf.reshape(x, [-1, 3, 3, 16])),

            k.layers.Conv2DTranspose(16, kernel_size=5, strides=4, padding="same"),
            LayerNormalizer(),
            k.layers.Activation(k.activations.relu),

            k.layers.Conv2DTranspose(12, kernel_size=3, strides=2, padding="same"),
            LayerNormalizer(),
            k.layers.Activation(k.activations.relu)
        ]
    

def main():
    # Generate Data
    pend_params = Pendulum.pendulum_default_params()
    pend_params[Pendulum.FRICTION_KEY] = 0.1
    data = Pendulum(24, observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                    transition_noise_std=0.1,
                    observation_noise_std=1e-5,
                    seed=0,
                    pendulum_params=pend_params)

    train_obs, train_obs_valid, train_targets = generate_imputation_data_set(data, 1000, 75, seed=42)
    valid_obs, valid_obs_valid, valid_targets = generate_imputation_data_set(data, 100, 75, seed=42)
    test_obs, test_obs_valid, test_targets = generate_imputation_data_set(data, 100, 75, seed=23541)

    # Build Model
    gin = PendulumImageImputationGIN(observation_shape=train_obs.shape[-3:], latent_observation_dim=8,
                                     output_dim=train_targets.shape[-3:], num_basis=15, never_invalid=False)

    # Train Model
    epochs, batch_size = 500,10
    Training_Loss = gin.training( gin, train_obs, train_obs_valid, train_targets, valid_obs, valid_obs_valid, valid_targets,
                                 epochs, batch_size)
    Test_Loss = gin.testing( gin, test_obs, test_obs_valid, test_targets, batch_size)

if __name__ == '__main__':
	main()
