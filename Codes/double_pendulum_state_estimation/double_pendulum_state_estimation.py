import numpy as np
from tensorflow import keras as k
from DoublePendulum import DoublePendulum
from GIN import GIN
from LayerNormalizer import LayerNormalizer


def generate_pendulum_filter_dataset( num_seqs, seq_length, seed):
    pendulum = DoublePendulum(num_seqs, seq_length, 24, seed)
    obs, targets = pendulum.datagen()
    obs, _ = pendulum.add_observation_noise(obs, first_n_clean=5, corr=0.2, lowlow=0.0, lowup=0.25, uplow=0.75, upup=1.0)
    obs = np.expand_dims(obs, -1)
    return obs.astype(np.float32), targets.astype(np.float32)


# Implement Encoder and Decoder hidden layers
class DoublePendulumStateEstemRKN(GIN):

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
    train_obs, train_targets = generate_pendulum_filter_dataset( 1000, 75, np.random.randint(100000000))
    valid_obs, valid_targets = generate_pendulum_filter_dataset( 100, 75, np.random.randint(100000000))
    test_obs, test_targets = generate_pendulum_filter_dataset( 100, 75, np.random.randint(10000000))


    # Build Model
    gin = DoublePendulumStateEstemRKN(observation_shape=train_obs.shape[-3:], latent_observation_dim=8, output_dim=4, num_basis=15,
                                never_invalid=True)


    # Train Model
    epochs, batch_size = 500,10
    Training_Loss = gin.training( gin, train_obs, train_targets, valid_obs, valid_targets,
                                 epochs, batch_size)
    Testing_Loss = gin.testing( gin, test_obs, test_targets,
                                 batch_size)
    
if __name__ == '__main__':
	main()

