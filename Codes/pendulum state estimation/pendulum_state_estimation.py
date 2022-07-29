import numpy as np
import tensorflow as tf
from tensorflow import keras as k
from PendulumData import Pendulum
from GIN import GIN
from LayerNormalizer import LayerNormalizer


def generate_pendulum_filter_dataset(pendulum, num_seqs, seq_length, seed):
    obs, targets, _, _ = pendulum.sample_data_set(num_seqs, seq_length, full_targets=False, seed=seed)
    obs, _ = pendulum.add_observation_noise(obs, first_n_clean=5, r=0.2, t_ll=0.0, t_lu=0.25, t_ul=0.75, t_uu=1.0)
    obs = np.expand_dims(obs, -1)
    return obs.astype(np.float32), targets.astype(np.float32)


# Implement Encoder and Decoder hidden layers
class PendulumStateEstemGIN(GIN):

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
        return [k.layers.Dense(units=10, activation=k.activations.relu)]

    def build_var_decoder_hidden(self):
        return [k.layers.Dense(units=10, activation=k.activations.relu)]
     
    


def main():
    # Generate Data
    pend_params = Pendulum.pendulum_default_params()
    pend_params[Pendulum.FRICTION_KEY] = 0.1
    data = Pendulum(24, observation_mode=Pendulum.OBSERVATION_MODE_LINE,
                    transition_noise_std=0.1,
                    observation_noise_std=1e-5,
                    seed=0,
                    pendulum_params=pend_params)

    train_obs, train_targets = generate_pendulum_filter_dataset(data, 1000, 75, np.random.randint(100000000))
    valid_obs, valid_targets = generate_pendulum_filter_dataset(data, 100, 75, np.random.randint(100000000))
    test_obs, test_targets = generate_pendulum_filter_dataset(data, 100, 75, np.random.randint(10000000))

    #Build Model
    gin = PendulumStateEstemGIN(observation_shape=train_obs.shape[-3:], latent_observation_dim=10, output_dim=2, num_basis=15,
                                 never_invalid=True)


    # Train Model
    epochs, batch_size = 500,10

    Training_Loss = gin.training( gin, train_obs, train_targets, valid_obs, valid_targets,
                                 epochs, batch_size)
    Test_Loss = gin.testing( gin, test_obs, test_targets, batch_size)

if __name__ == '__main__':
	main()
