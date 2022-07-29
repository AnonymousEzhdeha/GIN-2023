import tensorflow as tf
from tensorflow import keras as k
import numpy as np
from GINTransitionCell import GINTransitionCell, pack_input, unpack_state, pack_state


class GIN(k.models.Model):

    def __init__(self, observation_shape, latent_observation_dim, output_dim, 
                 never_invalid=False, cell_type="gin", state_ful = True, Qnetwork = "Xgru"):
        """
        :param observation_shape: shape of the observation to work with
        :param latent_observation_dim: latent observation dimension (m in paper)
        :param output_dim: dimensionality of model output
        :param never_invalid: if you know a-priori that the observation valid flag will always be positive you can set
                              this to true for slightly increased performance (obs_valid mask will be ignored)
        :param cell_type: type of cell to use "rkn" for our approach, "lstm" or "gru" for baselines
        :param state_ful: keep the state of the cell and pass them to the next cell (if any)
        """
        super().__init__()

        self._obs_shape = observation_shape
        self._lod = latent_observation_dim
        self._lsd = self._lod *2
        self._output_dim = output_dim
        self._never_invalid = never_invalid
        self._ld_output = np.isscalar(self._output_dim)
        self.cell_type = cell_type
        self.state_ful = state_ful

        
        self._layer_w_mean_norm = k.layers.TimeDistributed(k.layers.Lambda(
            lambda x: x / tf.norm(x, ord='euclidean', axis=-1, keepdims=True)))
        self._layer_w_covar = k.layers.TimeDistributed(
            k.layers.Dense(self._lod, activation=lambda x: k.activations.elu(x) + 1))

        # build transition
        if cell_type.lower() == "gin":
            self._cell = GINTransitionCell(self._lsd, self._lod,
                                           init_Q_matrices = 0.,
                                           init_KF_matrices = 0.,
                                           Qnetwork = Qnetwork,
                                           never_invalid=never_invalid)
        elif cell_type.lower() == "lstm":
            print("Running LSTM Baseline")
            self._cell = k.layers.LSTMCell(2 * self._lsd)
        elif cell_type.lower() == "gru":
            print("Running GRU Baseline")
            self._cell = k.layers.GRUCell(2 * self._lsd)
        else:
            raise AssertionError("Invalid Cell type, needs tp be 'gin', 'lstm' or 'gru'")

        self._layer_rkn = k.layers.RNN(self._cell, return_sequences=True, stateful=self.state_ful)

        
        if self._ld_output:
            # build decoder mean
            self._layer_dec_out = k.layers.TimeDistributed(k.layers.Dense(units=self._output_dim))

            # build decoder variance
            self._var_dec_hidden = self._time_distribute_layers(self.build_var_decoder_hidden())
            self._layer_var_dec_out = k.layers.TimeDistributed(
                k.layers.Dense(units=self._output_dim, activation=lambda x: k.activations.elu(x) + 1))

        else:
            self._layer_dec_out = k.layers.TimeDistributed(
                k.layers.Conv2DTranspose(self._output_dim[-1], kernel_size=3, padding="same",
                                         activation=k.activations.sigmoid))

        
    def build_decoder_hidden(self):
        """
        Implement mean decoder hidden layers
        :return: list of mean decoder hidden layers
        """
        raise NotImplementedError

    def build_var_decoder_hidden(self):
        """
        Implement var decoder hidden layers
        :return: list of var decoder hidden layers
        """
        raise NotImplementedError

    def call(self, inputs, initialization=False, training=None, mask=None):
        """
        :param inputs: model inputs (i.e. observations)
        :param training: required by k.models.Models
        :param mask: required by k.models.Model
        :return:
        """
        H = np.array([[1., 0, 0, 0],[0, 0, 1, 0]])    
        Init_Hmatrix = np.tile( np.expand_dims(np.array(np.array(H).astype(np.float32)),0) ,
                                [inputs.shape[0], 1, 1]) 
        self.H_matrix = tf.constant(Init_Hmatrix, dtype = tf.float32)
        
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            img_inputs, obs_valid = inputs
        else:
            assert self._never_invalid, "If invalid inputs are possible, obs_valid mask needs to be provided"
            img_inputs = inputs
            obs_valid = tf.ones([tf.shape(img_inputs)[0], tf.shape(img_inputs)[1], 1])

        # observation mean and covar
        w_mean = img_inputs
        w_covar = self._layer_w_covar(img_inputs)
        

        # transition
        rkn_in = pack_input(w_mean, w_covar, obs_valid)
        if self.cell_type.lower() == 'gin':
            # initial states to the first observed data if we are at the first stage, otherwise use 'stateful'
            # to obtain state from previous update
            
            if initialization:
                Init_covar = np.tile( np.expand_dims(np.array(1 * np.eye(self._lsd, self._lsd).astype(np.float32)),0) ,
                                    [inputs.shape[0], 1, 1]) 
                Init_covar = np.reshape(Init_covar, (Init_covar.shape[0], -1))
                initial_covar = tf.constant(Init_covar)
                initial_mean = tf.expand_dims( tf.constant([w_mean[0][0][0].numpy(), 0., w_mean[0][0][1].numpy(), 0.]), axis=0)
                init_state =  pack_state(initial_mean, initial_covar)
                z = self._layer_rkn(rkn_in, initial_state = init_state)
            else:
                z = self._layer_rkn(rkn_in)
            post_mean, post_covar = unpack_state(z, self._lsd)
            post_covar = tf.concat(post_covar, -1)

            # if self.Smoothing:
            #     smooth_mean_init, smooth_covar_init, post_mean_reverse, post_covar_reverse, prior_mean_reverse, prior_covar_reverse, transition_matrix_reverse = self.z_time_reverse(z)
            #     init_state = pack_state(smooth_mean_init, smooth_covar_init)
            #     post_mean_reverse, post_covar_reverse = self._layer_smooth((post_mean_reverse, post_covar_reverse, prior_mean_reverse, prior_covar_reverse,
            #                                                 transition_matrix_reverse), initial_state = init_state)
            #     post_mean_reverse = tf.concat([tf.expand_dims(smooth_mean_init, axis=1), post_mean_reverse], axis=1)
            #     post_covar_reverse = tf.concat([tf.expand_dims(smooth_covar_init, axis=1), post_covar_reverse], axis=1)
            #     post_mean = tf.reverse(post_mean_reverse, axis=[1])
            #     post_covar = tf.reverse(post_covar_reverse, axis =[1])
            #     post_covar = tf.concat(post_covar, -1)

        else:
            z = self._layer_rkn(rkn_in)
            post_mean, post_covar = unpack_state(z, self._lsd)
            post_covar = tf.concat(post_covar, -1)

        # decode
        pred_mean = tf.transpose( tf.matmul(self.H_matrix, tf.transpose(post_mean, perm = [0,2,1])), perm=[0,2,1] )
        if self._ld_output:
            pred_var = self._layer_var_dec_out(post_covar)
            return tf.concat([pred_mean, pred_var], -1)
        else:
            return pred_mean
        
        

    # loss functions
    def gaussian_nll(self, target, pred_mean_var):
        """
        gaussian nll
        :param target: ground truth positions
        :param pred_mean_var: mean and covar (as concatenated vector, as provided by model)
        :return: gaussian negative log-likelihood
        """
        pred_mean, pred_var = pred_mean_var[..., :self._output_dim], pred_mean_var[..., self._output_dim:]
        pred_var += 1e-8
        element_wise_nll = 0.5 * (np.log(2 * np.pi) + tf.math.log(pred_var) + ((target - pred_mean)**2) / pred_var)
        sample_wise_error = tf.reduce_sum(element_wise_nll, axis=-1)
        sample_wise_var = tf.reduce_sum(tf.math.log(pred_var), axis=-1)
        return tf.reduce_mean(sample_wise_error), tf.reduce_mean(sample_wise_var)

    def rmse(self, target, pred_mean_var):
        """
        root mean squared error
        :param target: ground truth positions
        :param pred_mean_var: mean and covar (as concatenated vector, as provided by model)
        :return: root mean squared error between targets and predicted mean, predicted variance is ignored
        """
        pred_mean = pred_mean_var[..., :self._output_dim]
        return tf.sqrt(tf.reduce_mean((pred_mean - target) ** 2))
    
    def rmse_batch(self, target, pred_mean_var, batch_size):
        """
        root mean squared error for batch, calculate the loss for each element
        :param target: ground truth positions
        :param pred_mean_var: mean and covar (as concatenated vector, as provided by model)
        :return: root mean squared error  of single element between targets and predicted mean, predicted variance is ignored
        """
        pred_mean = pred_mean_var[..., :self._output_dim]
        return tf.sqrt(tf.reduce_mean((pred_mean - target) ** 2))/ batch_size

    def bernoulli_nll(self, targets, predictions, uint8_targets=True):
        """ Computes Binary Cross Entropy
        :param targets:
        :param predictions:
        :param uint8_targets: if true it is assumed that the targets are given in uint8 (i.e. the values are integers
        between 0 and 255), thus they are devided by 255 to get "float image representation"
        :return: Binary Crossentropy between targets and prediction
        """
        if uint8_targets:
            targets = targets / 255.0
        point_wise_error = - (
                    targets * tf.math.log(predictions + 1e-12) + (1 - targets) * tf.math.log(1 - predictions + 1e-12))
        red_axis = [i + 2 for i in range(len(targets.shape) - 2)]
        sample_wise_error = tf.reduce_sum(point_wise_error, axis=red_axis)
        return tf.reduce_mean(sample_wise_error)
    
    def _MSE(self, Qpreds, targets):
          
          return tf.math.reduce_mean(tf.square(Qpreds - targets))

    def training(self, model, Train_Obs, Train_Target, Valid_Obs, Valid_Target, epochs, batch_size=1):
        
        ##val_batching
        T = 10
        Ybatch_val = []
        Ubatch_val = []
        
        for bid in range(int(len(Valid_Target)/batch_size)):
            Ybatch_val.append( Valid_Target[bid*batch_size:(bid+1)*batch_size])
        for bid in range(int(len(Valid_Obs)/batch_size)):
            Ubatch_val.append( Valid_Obs[bid*batch_size:(bid+1)*batch_size])
        
        Ybatch_val = np.array(Ybatch_val)
        Ubatch_val = np.array(Ubatch_val)
        
        ##train_batching
        batch_size = batch_size
        Ybatch = []
        Ubatch = []
        
        for bid in range(int(len(Train_Target)/batch_size)):
            Ybatch.append( Train_Target[bid*batch_size:(bid+1)*batch_size])
        for bid in range(int(len(Train_Obs)/batch_size)):
            Ubatch.append( Train_Obs[bid*batch_size:(bid+1)*batch_size])
        
        Ybatch = np.array(Ybatch)
        Ubatch = np.array(Ubatch)
        
        Training_Loss = []
        for epoch in range(epochs):
            initialization = True
            loss_show_mse_tr = 0.
            for i in range(len(Ybatch)):
                if (i%T ==0) and ( i!=0):
                    initialization = True
                    self._layer_rkn.reset_states()
                # NetIn = tf.expand_dims(Train_Obs[:10], axis=0)
                NetIn = Ubatch[i]
                with tf.GradientTape() as tape:
                    preds = model(NetIn, initialization)
                    loss_mse = self.rmse(Ybatch[i], preds)
                    loss_show_mse_tr += loss_mse
                variables = model.trainable_variables
                
                
                gradients = tape.gradient(loss_mse, variables)
                tf.keras.optimizers.Adam(clipnorm=5.0).apply_gradients(zip(gradients, variables))

                if i %T==0:
                    rand_sel = np.random.randint(0, len(Ubatch_val))
                    val_preds = model(Ubatch_val[rand_sel])
                    print('val loss %s' % (self.rmse_batch(Ybatch_val[rand_sel], val_preds, batch_size).numpy()))
                
                if (i%T==0) and(i !=0):
                    print('epoch %d  loss %s' % (epoch, loss_show_mse_tr.numpy() ))
                    Training_Loss.append(loss_show_mse_tr/batch_size)
                    loss_show_mse_tr = 0.
                    
                initialization = False
            self._layer_rkn.reset_states()
            # self._layer_rkn.reset_states()
        return Training_Loss
    

    def testing(self, model, test_obs, test_targets, batch_size=1):
        T=10
        batch_size = batch_size
        Ybatch = []
        Ubatch = []
        
        for bid in range(int(len(test_targets)/batch_size)):
            Ybatch.append( test_targets[bid*batch_size:(bid+1)*batch_size])
        for bid in range(int(len(test_obs)/batch_size)):
            Ubatch.append( test_obs[bid*batch_size:(bid+1)*batch_size])        
        Ybatch = np.array(Ybatch)
        Ubatch = np.array(Ubatch)

        Test_Loss = []
        Test_loss_accumulate = []
        self._layer_rkn.reset_states()
        initialization = True
        loss_mse_accumulative = 0.
        for i in range(len(Ybatch)):
            if (i%T ==0) and(i != 0):
                initialization = True
                self._layer_rkn.reset_states()
            NetIn = Ubatch[i]
            preds = model(NetIn, initialization)
            loss = self.rmse_batch(Ybatch[i], preds, batch_size)
            loss_mse_accumulative += loss
            print('test loss: %s' % (loss))
            if (i%T==0):
                print('test loss accumulative: %s' % (loss_mse_accumulative))
                Test_loss_accumulate.append(loss_mse_accumulative.numpy())
                loss_mse_accumulative = 0.
            Test_Loss.append(loss.numpy())
            initialization = False
        print('accumulative test_loss %s' % (tf.reduce_mean(Test_loss_accumulate)))
        return Test_Loss

    
    @staticmethod
    def _prop_through_layers(inputs, layers):
        """propagates inputs through layers"""
        h = inputs
        for layer in layers:
            h = layer(h)
        return h

    @staticmethod
    def _time_distribute_layers(layers):
        """wraps layers with k.layers.TimeDistributed"""
        td_layers = []
        for l in layers:
            td_layers.append(k.layers.TimeDistributed(l))
        return td_layers