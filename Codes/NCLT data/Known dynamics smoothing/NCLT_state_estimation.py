import numpy as np
from tensorflow import keras as k
import NCLT_data
from GIN import GIN
import math


def split2(data, split_size):
    splited_data = []
    for dt in data:
        dt = np.reshape(dt[:(math.floor(dt.shape[0] / split_size) ) *  split_size], (-1, split_size, dt.shape[-1]))
        splited_data.append( dt )
    return splited_data

# Implement Encoder and Decoder hidden layers
class NCLTStateEstemGIN(GIN):

    def build_decoder_hidden(self):
        return [k.layers.Dense(units=3, activation=k.activations.relu)]

    def build_var_decoder_hidden(self):
        return [k.layers.Dense(units=3, activation=k.activations.relu)]
      


def main():
     
    split_size2 = 1
    
    ##data Length and Batch_Size
    #train_obs, train_targets, test_obs, test_targets, valid_obs, valid_targets = Generate_Data(split_size2)
    train_obs, train_targets, test_obs, test_targets, valid_obs, valid_targets = NCLT_data.NCLT_DG(split_size2)
    
    data = [train_obs, train_targets, test_obs, test_targets, valid_obs, valid_targets]
    # sp_train_obs, sp_train_targets, sp_test_obs, sp_test_targets, sp_valid_obs, sp_valid_targets = split(data, split_size)
    
    sp_train_obs, sp_train_targets, sp_test_obs, sp_test_targets, sp_valid_obs, sp_valid_targets = split2(data, split_size2)
    
    NCLT = NCLTStateEstemGIN(observation_shape=train_obs.shape[-1], latent_observation_dim=2, output_dim=2,
                                    never_invalid=True)
    
    epochs = 100
    Training_Loss = NCLT.training( NCLT, sp_train_obs, sp_train_targets,
                                  sp_valid_obs, sp_valid_targets, epochs)
    Test_Loss = NCLT.testing( NCLT, sp_test_obs, sp_test_targets)
    ###

if __name__ == '__main__':
	main()

