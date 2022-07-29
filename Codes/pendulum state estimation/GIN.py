import tensorflow as tf
from tensorflow import keras as k
import numpy as np
from GINTransitionCell import GINTransitionCell, pack_input, unpack_state, pack_state
from GINSmoothCell import GINSmoothingCell

class GIN(k.models.Model):

    def __init__(self, observation_shape, latent_observation_dim, output_dim, num_basis,
                 trans_net_hidden_units=[], never_invalid=False, cell_type="gin", Qnetwork="nothing", USE_CONV=False, Smoothing = False):
        """
        obs_shape: shape of the observation 
        obs_dim: latent observation dimension 
        out_dim: dimensionality of model output
        num_basis: number of basis matrices 
        trans_net_hidden_units: hidden units for dynamics network
        never_invalid: boolean indicating whether all observations are available or a part of it is missing
        cell_type: type of cell to use "gin", "lstm" or "gru" 
        Qnetwork: defines the type of inference for Q matrix. "Xmlp", "Xgru", "Fmlp", "Fgru" and "nothing". 
            "Xmlp": Q = MLP(X^+)
            "Xgru": Q = GRU(X^+)
            "Fmlp": Q = MLP(F)
            "Fgru": Q = GRU(F)
            "nothing": Q is learned jointly with the transition matrix (F(Q) in the paper)
        USE_CONV: defines whether use the convolutional layer for the covariance matrix or not
        """
        super().__init__()

        self._obs_shape = observation_shape
        self._lod = latent_observation_dim
        self._lsd = 2 * self._lod
        self._output_dim = output_dim
        self._never_invalid = never_invalid
        self._ld_output = np.isscalar(self._output_dim)
        self.cell_type = cell_type
        self.Smoothing = Smoothing
        # build encoder
        self._enc_hidden_layers = self._time_distribute_layers(self.build_encoder_hidden())

        # we need to ensure the bias is initialized with non-zero values to ensure the normalization does not produce
        # nan
        self._layer_w_mean = k.layers.TimeDistributed(
            k.layers.Dense(self._lod, activation=k.activations.linear,
                           bias_initializer=k.initializers.RandomNormal(stddev=0.05)))
        self._layer_w_mean_norm = k.layers.TimeDistributed(k.layers.Lambda(
            lambda x: x / tf.norm(x, ord='euclidean', axis=-1, keepdims=True)))
        self._layer_w_covar = k.layers.TimeDistributed(
            k.layers.Dense(self._lod, activation=lambda x: k.activations.elu(x) + 1))

        # build transition
        if cell_type.lower() == "gin":
            self._cell = GINTransitionCell(self._lsd, self._lod,
                                           number_of_basis=num_basis,
                                           init_kf_matrices=0.05,
                                           init_Q_matrices = 0.05,
                                           init_KF_matrices = 0.1,
                                           Qnetwork=Qnetwork,
                                           USE_CONV = USE_CONV,
                                           trans_net_hidden_units=trans_net_hidden_units,
                                           never_invalid=never_invalid)
        elif cell_type.lower() == "lstm":
            print("Running LSTM Baseline")
            self._cell = k.layers.LSTMCell(2 * self._lsd)
        elif cell_type.lower() == "gru":
            print("Running GRU Baseline")
            self._cell = k.layers.GRUCell(2 * self._lsd)
        else:
            raise AssertionError("Invalid Cell type, needs tp be 'gin', 'lstm' or 'gru'")

        self._layer_rkn = k.layers.RNN(self._cell, return_sequences=True)
        if self.Smoothing:
            self._smoothing_cell = GINSmoothingCell(self._lsd,
                                                    self._lod,
                                                    init_kf_matrices = 0.05,
                                                    init_KF_matrices = 0.05,
                                                    USE_CONV = USE_CONV)
            self._layer_smooth = k.layers.RNN(self._smoothing_cell, return_sequences=True)

        self._dec_hidden = self._time_distribute_layers(self.build_decoder_hidden())
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



    def build_encoder_hidden(self):
        """
        if required, it would be built in a subclass of th GIN
        """
        raise NotImplementedError

    def build_decoder_hidden(self):
        """
        if required, it would be built in a subclass of th GIN
        """
        raise NotImplementedError

    def build_var_decoder_hidden(self):
        """
        if required, it would be built in a subclass of th GIN
        """
        raise NotImplementedError

    def call(self, inputs, training=None, mask=None):
        """
        inputs: original observations
        training: required by k.models.Models
        mask: required by k.models.Model
        
        """
        if isinstance(inputs, tuple) or isinstance(inputs, list):
            img_inputs, obs_valid = inputs
        else:
            assert self._never_invalid, "If invalid inputs are possible, obs_valid mask needs to be provided"
            img_inputs = inputs
            obs_valid = tf.ones([tf.shape(img_inputs)[0], tf.shape(img_inputs)[1], 1])

        # encoder
        enc_last_hidden = self._prop_through_layers(img_inputs, self._enc_hidden_layers)
        w_mean = self._layer_w_mean_norm(self._layer_w_mean(enc_last_hidden))
        w_covar = self._layer_w_covar(enc_last_hidden)
        
        # transition
        rkn_in = pack_input(w_mean, w_covar, obs_valid)
        z = self._layer_rkn(rkn_in)

        if self.cell_type.lower() == 'gin':
            # unpack outputs;[ post_mean = mu_t|t, (posterior_mean, i.e. mean filtered),
                            # post_covar = sigma_t|t, (posterior_covar, i.e. covar filtered) 
                            # prior_mean = mu_t|t-1 = A_t mu_t-1|t-1, 
                            # prior_covar = sigma_t|t-1 = A_t sigma_t-1|t-1 A_t^T + Q_t
                            # transition_matrix = A_t
            post_mean, post_covar, prior_mean, prior_covar, transition_matrix = z
            post_covar = tf.concat(post_covar, -1)
            if self.Smoothing:
                smooth_mean_init, smooth_covar_init, post_mean_reverse, post_covar_reverse, prior_mean_reverse, prior_covar_reverse, transition_matrix_reverse = self.z_time_reverse(z)
                init_state = pack_state(smooth_mean_init, smooth_covar_init)
                post_mean_reverse, post_covar_reverse = self._layer_smooth((post_mean_reverse, post_covar_reverse, prior_mean_reverse, prior_covar_reverse,
                                                            transition_matrix_reverse), initial_state = init_state)
                post_mean_reverse = tf.concat([tf.expand_dims(smooth_mean_init, axis=1), post_mean_reverse], axis=1)
                post_covar_reverse = tf.concat([tf.expand_dims(smooth_covar_init, axis=1), post_covar_reverse], axis=1)
                post_mean = tf.reverse(post_mean_reverse, axis=[1])
                post_covar = tf.reverse(post_covar_reverse, axis =[1])
                post_covar = tf.concat(post_covar, -1)

        else:
            post_mean, post_covar = unpack_state(z, self._lsd)
            post_covar = tf.concat(post_covar, -1)

        # decode
        pred_mean = self._layer_dec_out(self._prop_through_layers(post_mean, self._dec_hidden))
        if self._ld_output:
            pred_var = self._layer_var_dec_out(self._prop_through_layers(post_covar, self._var_dec_hidden))
            return tf.concat([pred_mean, pred_var], -1)
        else:
            return pred_mean
    def z_time_reverse(self, z):
        post_mean, post_covar, prior_mean, prior_covar, transition_matrix = z
        smooth_mean_init = post_mean[:, -1, :]
        smooth_covar_init = post_covar[:, -1, :]

        post_mean_reverse = tf.reverse(post_mean[:, :-1, :], [1])
        post_covar_reverse = tf.reverse(post_covar[:, :-1, :], [1])
        prior_mean_reverse = tf.reverse(prior_mean[:, 1:, :], [1])
        prior_covar_reverse = tf.reverse(prior_covar[:, 1:, :], [1])
        transition_matrix_reverse = tf.reverse(transition_matrix[:, 1:, :, :], [1])
        return smooth_mean_init, smooth_covar_init, post_mean_reverse, post_covar_reverse, prior_mean_reverse, prior_covar_reverse, transition_matrix_reverse
    # loss functions
    def gaussian_nll(self, target, pred_mean_var):
        """
        output with gaussian assumption distribution
        target: ground truth
        pred_mean_var: mean and covar 
        
        """
        pred_mean, pred_var = pred_mean_var[..., :self._output_dim], pred_mean_var[..., self._output_dim:]
        pred_var += 1e-8
        element_wise_nll = 0.5 * (np.math.log(2 * np.pi) + tf.math.log(pred_var) + ((target - pred_mean)**2) / pred_var)
        sample_wise_error = tf.reduce_sum(element_wise_nll, axis=-1)
        return tf.reduce_mean(sample_wise_error)

    def rmse(self, target, pred_mean_var):
        """
        MSE
        target: ground truth 
        pred_mean_var: mean and covar 
        
        """
        pred_mean = pred_mean_var[..., :self._output_dim]
        return tf.sqrt(tf.reduce_mean((pred_mean - target) ** 2))

    def bernoulli_nll(self, targets, predictions, uint8_targets=True):
        """ 
        output with bernoulli assumption distribution
        targets: ground truth
        predictions: model's prediction
        uint8_targets: if true it is assumed that the targets are given in uint8 
        
        """
        if uint8_targets:
            targets = targets / 255.0
        point_wise_error = - (
                    targets * tf.math.log(predictions + 1e-12) + (1 - targets) * tf.math.log(1 - predictions + 1e-12))
        red_axis = [i + 2 for i in range(len(targets.shape) - 2)]
        sample_wise_error = tf.reduce_sum(point_wise_error, axis=red_axis)
        return tf.reduce_mean(sample_wise_error)
    
    def training(self, model, Train_Obs, Train_Target, Valid_Obs, Valid_Target, epochs, batch_size):
        
        """
        training procedure
        depending on the task, appropriate loss function is taken account
        """
        
        ##val_batching
        Ybatch_val = []
        Ubatch_val = []
        
        for bid in range(int(len(Valid_Target)/batch_size)):
            Ybatch_val.append( Valid_Target[bid*batch_size:(bid+1)*batch_size])
        for bid in range(int(len(Valid_Obs)/batch_size)):
            Ubatch_val.append( Valid_Obs[bid*batch_size:(bid+1)*batch_size])
        
        Ybatch_val = np.array(Ybatch_val)
        Ubatch_val = np.array(Ubatch_val)
        
        #train_batching
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
            for i in range(len(Ybatch)):
                # NetIn = tf.expand_dims(Train_Obs[:10], axis=0)
                NetIn = Ubatch[i]
                with tf.GradientTape() as tape:
                    preds = model(NetIn)
                    loss = self.gaussian_nll(Ybatch[i], preds)

                variables = model.trainable_variables
                gradients = tape.gradient(loss, variables)
                tf.keras.optimizers.Adam(clipnorm=5.0).apply_gradients(zip(gradients, variables))
                if i %10==0:
                    rand_sel = np.random.randint(0, len(Ubatch_val))
                    val_preds = model(Ubatch_val[rand_sel])
                    val_loss = self.gaussian_nll(Ybatch_val[rand_sel], val_preds)
                    print('val loss: %s' % (val_loss.numpy()))
                
                print('epoch: %d  loss_gaussian: %s' % (epoch, loss.numpy()))
                Training_Loss.append(loss)  
        return Training_Loss
    
    def testing(self, model, test_obs, test_targets, batch_size):
        """
        testing procedure
        depending on the task, appropriate loss function is taken account
        """
        
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
        for i in range(len(Ybatch)):
            NetIn = Ubatch[i]
            preds = model(NetIn)
            loss = self.gaussian_nll(Ybatch[i], preds)
            print(' test_loss: %s' % (loss.numpy()))
            Test_Loss.append(loss)
        
        return Test_Loss

    # helpers
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