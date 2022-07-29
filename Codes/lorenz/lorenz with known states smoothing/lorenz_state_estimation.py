import numpy as np
from tensorflow import keras as k
from LorenzSysModel import SystemModel
from parameters import m, n
import model
from GIN import GIN



def Generate_Data(num_seqs_train=1, num_seqs_test=1, num_seqs_valid=1, seq_length_train=1, seq_length_test=1, seq_length_valid=1, q=1, r=1):
    
    obj_sysmodel = SystemModel( model.f, q, model.h, r, seq_length_train,  m, n)
    train_obs, train_targets = obj_sysmodel.GenerateBatch(num_seqs_train, seq_length_train, randomInit=True)
    obj_sysmodel = SystemModel( model.f, q, model.h, r, seq_length_test,  m, n)
    test_obs, test_targets = obj_sysmodel.GenerateBatch(num_seqs_test, seq_length_test, randomInit=True)
    obj_sysmodel = SystemModel( model.f, q, model.h, r, seq_length_valid,  m, n)
    valid_obs, valid_targets = obj_sysmodel.GenerateBatch(num_seqs_valid, seq_length_valid, randomInit=True)

    #reorder data to #(num of samples, length, features)
    train_obs = np.transpose(train_obs.numpy(), axes=[0,2,1]) 
    train_targets = np.transpose(train_targets.numpy(), axes=[0,2,1])
    test_obs = np.transpose(test_obs.numpy(), axes=[0,2,1])
    test_targets = np.transpose(test_targets.numpy(), axes=[0,2,1])
    valid_obs = np.transpose(valid_obs.numpy(), axes=[0,2,1])
    valid_targets = np.transpose(valid_targets.numpy(), axes=[0,2,1])
    return train_obs, train_targets, test_obs, test_targets, valid_obs, valid_targets

def split(data, split_size):
    splited_data = []
    for dt in data:
        dt = np.reshape(dt, ( split_size,-1, dt.shape[-1]), order= 'F')
        splited_data.append( np.transpose(dt, axes=[1,0,2]) )
    return splited_data

# Implement Encoder and Decoder hidden layers
class LorenzStateEstemGIN(GIN):

    def build_decoder_hidden(self):
        return [k.layers.Dense(units=3, activation=k.activations.relu)]

    def build_var_decoder_hidden(self):
        return [k.layers.Dense(units=3, activation=k.activations.relu)]
      


def main():
    # Generate observation and measurement parameters (i.e sigma_r^2I and sigma_q^2I) 
    r2 = 0.25 ### r^2 = 0.25
    r = np.sqrt(r2)
    vdB = -20 # ratio v=q2/r2
    v = 10**(vdB/10)
    q2 = v*r2
    q = np.sqrt(q2) 
    split_size = 1
    
    ##data Length and Batch_Size
    train_samples = [50, 100, 200, 500, 1000, 2000, 5000, 10000, 20000, 50000, 100000]
    
    MSE_TEST = []
    
    for length_train in train_samples:
        ##data gen
        train_obs, train_targets, test_obs, test_targets, valid_obs, valid_targets = Generate_Data(num_seqs_train=1, num_seqs_test=1,
                               num_seqs_valid=1, seq_length_train=length_train, seq_length_test=400, seq_length_valid=400, q=q, r=r)
        data = [train_obs, train_targets, test_obs, test_targets, valid_obs, valid_targets]
        sp_train_obs, sp_train_targets, sp_test_obs, sp_test_targets, sp_valid_obs, sp_valid_targets = split(data, split_size)

        ##
        ## Build Model
        Lorenz = LorenzStateEstemGIN(observation_shape=train_obs.shape[-1], latent_observation_dim=3, output_dim=3,
                                   never_invalid=True)


        # # Train Model
        epochs = 100
        Training_Loss = Lorenz.training( Lorenz, sp_train_obs, sp_train_targets,
                                     sp_valid_obs, sp_valid_targets, epochs)
        Test_Loss = Lorenz.testing( Lorenz, sp_test_obs, sp_test_targets)
    MSE_TEST.append(Test_Loss)

if __name__ == '__main__':
	main()

