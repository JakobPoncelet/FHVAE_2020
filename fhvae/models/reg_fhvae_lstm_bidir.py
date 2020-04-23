import tensorflow as tf
from tensorflow.keras import layers
import numpy as np

class RegFHVAE_bidirectional(tf.keras.Model):
    ''' combine the encoder and decoder into an end-to-end model for training
        uses bidirectional LSTMs for the encoders and the decoder '''

    def __init__(self, z1_dim=32, z2_dim=32, z1_rhus=[256, 256], z2_rhus=[256, 256], x_rhus=[256, 256],
                 nmu2=5000, z1_nlabs={}, z2_nlabs={}, mu_nl=None,logvar_nl=None, tr_shape=(20, 80),
                 bs=256, alpha_dis_z1=1.0, alpha_dis_z2=1.0, alpha_reg_b=1.0, alpha_reg_c=1.0, n_phones=62, name="autoencoder", **kwargs):

        super(RegFHVAE_bidirectional, self).__init__(name=name, **kwargs)

        # input shape
        self.tr_shape = tr_shape
        self.bs = bs

        # encoder/decoder arch
        self.z1_dim, self.z2_dim = z1_dim, z2_dim
        self.z1_rhus, self.z2_rhus = z1_rhus, z2_rhus
        self.x_rhus = x_rhus

        # non linearities
        self.mu_nl, self.logvar_nl = mu_nl, logvar_nl

        # nlabs = dictionary with for each label, the dimension of the regularization vector
        self.z1_nlabs, self.z2_nlabs = z1_nlabs, z2_nlabs
        self.nmu2 = nmu2
        self.mu2_table = tf.Variable(tf.random.normal([nmu2, z2_dim], stddev=1.0), trainable=False)

        # Trainable=False to remove warnings about tf.gather()

        self.n_phones = n_phones
        self.mu1_table = tf.Variable(tf.random.normal([n_phones, z1_dim], stddev=1.0), trainable=False)
        self.phone_occs = tf.Variable(tf.zeros([n_phones]), trainable=False)

        # loss factors
        self.alpha_dis_z1, self.alpha_dis_z2 = alpha_dis_z1, alpha_dis_z2
        self.alpha_reg_b, self.alpha_reg_c = alpha_reg_b, alpha_reg_c

        # init net
        self.encoder = Encoder(self.z1_dim, self.z2_dim, self.z1_rhus, self.z2_rhus, self.tr_shape, self.mu_nl, self.logvar_nl)
        self.decoder = Decoder(self.x_rhus, self.tr_shape, self.mu_nl, self.logvar_nl)
        self.regulariser = Regulariser(self.z1_nlabs, self.z2_nlabs)

        # log-prior stddevs
        # self.pz1_stddev = 1.0
        self.pz1_stddev = 0.5
        self.pmu1_stddev = 1.0
        self.pz2_stddev = 0.5
        self.pmu2_stddev = 1.0

    def call(self, x, y, b):

        # lookup mu1 from table using talab
        mu1 = tf.gather(self.mu1_table, b)

        # lookup mu2 from table
        mu2 = tf.gather(self.mu2_table, y)

        z1_mu, z1_logvar, z1_sample, z2_mu, z2_logvar, z2_sample, qz1_x, qz2_x = self.encoder(x)

        out, x_mu, x_logvar, x_sample, px_z = self.decoder(x, z1_sample, z2_sample)

        z1_rlogits, z2_rlogits = self.regulariser(z1_mu, z2_mu)

        return mu2, mu1, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits

    def compute_loss(self, x, y, n, bReg, cReg, mu2, mu1, qz2_x, z2_sample, qz1_x, z1_sample, px_z, x_sample, z1_rlogits, z2_rlogits, num_seqs):

        fn = tf.nn.sparse_softmax_cross_entropy_with_logits

        # priors
        pz1 = [mu1, np.log(self.pz1_stddev ** 2).astype(np.float32)]
        pmu1 = [0., np.log(self.pmu1_stddev ** 2).astype(np.float32)]
        pz2 = [mu2, np.log(self.pz2_stddev ** 2).astype(np.float32)]
        pmu2 = [0., np.log(self.pmu2_stddev ** 2).astype(np.float32)]

        # variational lower bound
        log_pmu1 = log_normal_pdf(mu1, pmu1[0], pmu1[1], raxis=1)
        log_pmu1 = log_pmu1 / (n * num_seqs)
        log_pmu2 = log_normal_pdf(mu2, pmu2[0], pmu2[1], raxis=1)
        log_pmu2 = log_pmu2 / n
        log_px_z = log_normal_pdf(x, px_z[0], px_z[1], raxis=(1, 2))
        neg_kld_z1 = -1 * tf.reduce_sum(kld(qz1_x[0], qz1_x[1], pz1[0], pz1[1]), axis=1)
        neg_kld_z2 = -1 * tf.reduce_sum(kld(qz2_x[0], qz2_x[1], pz2[0], pz2[1]), axis=1)

        lb = log_px_z + neg_kld_z1 + neg_kld_z2 + log_pmu2 + log_pmu1

        # discriminative loss mu2
        logits = tf.expand_dims(qz2_x[0], 1) - tf.expand_dims(self.mu2_table, 0)
        logits = -1 * tf.pow(logits, 2) / (2 * tf.exp(pz2[1]))
        logits = tf.reduce_sum(input_tensor=logits, axis=-1)
        log_qy_mu2 = -fn(labels=y, logits=logits)

        # discriminative loss mu1
        logits = tf.expand_dims(qz1_x[0], 1) - tf.expand_dims(self.mu1_table, 0)
        logits = -1 * tf.pow(logits, 2) / (2 * tf.exp(pz1[1]))
        logits = tf.reduce_sum(input_tensor=logits, axis=-1)
        ## frequency of phonemes, +1 to not divide by 0
        logits = tf.divide(logits, self.phone_occs + 1)
        log_qy_mu1 = -fn(labels=bReg[:, 0], logits=logits)

        # Regularization loss
        TensorList = [
            tf.expand_dims(-fn(labels=tf.squeeze(tf.slice(bReg, [0, i], [-1, 1]), axis=-1), logits=z1_rlogits[i]),
                           axis=1) for i, name in enumerate(self.z1_nlabs.keys())]
        log_b = tf.concat(TensorList, axis=1)

        TensorList = [
            tf.expand_dims(-fn(labels=tf.squeeze(tf.slice(cReg, [0, i], [-1, 1]), axis=-1), logits=z2_rlogits[i]),
                           axis=1) for i, name in enumerate(self.z2_nlabs.keys())]
        log_c = tf.concat(TensorList, axis=1)

        log_b_loss = tf.reduce_sum(input_tensor=log_b, axis=1)
        log_c_loss = tf.reduce_sum(input_tensor=log_c, axis=1)

        # total loss
        loss = -1 * tf.reduce_mean(
            input_tensor=lb + self.alpha_dis_z2 * log_qy_mu2 + self.alpha_dis_z1 * log_qy_mu1 + self.alpha_reg_b * log_b_loss + self.alpha_reg_c * log_c_loss)

        return loss, log_pmu2, log_pmu1, neg_kld_z2, neg_kld_z1, log_px_z, lb, log_qy_mu2, log_qy_mu1, log_b_loss, log_c_loss


class Encoder(layers.Layer):
    ''' encodes the input to the latent factors z1 and z2'''

    def __init__(self, z1_dim=32, z2_dim=32, z1_rhus=[256,256], z2_rhus=[256,256],
                 tr_shape=(20,80), mu_nl=None, logvar_nl=None, name="encoder", **kwargs):

        super(Encoder, self).__init__(name=name,   **kwargs)

        # latent dims
        self.z1_dim = z1_dim
        self.z2_dim = z2_dim

        # RNN specs for z2_pre_encoder
        self.z2_rhus = z2_rhus

        ## Bidirectional LSTMs
        self.lstm_layer1_z2 = layers.Bidirectional(layers.LSTM(self.z2_rhus[0], return_sequences=True, return_state=True, time_major=False), merge_mode='concat')
        self.lstm_layer2_z2 = layers.Bidirectional(layers.LSTM(self.z2_rhus[1], return_state=True, time_major=False), merge_mode='concat')

        # RNN specs for z1_pre_encoder
        self.z1_rhus = z1_rhus

        ## Bidirectional LSTMs
        self.lstm_layer1_z1 = layers.Bidirectional(layers.LSTM(self.z1_rhus[0], return_sequences=True, return_state=True, time_major=False), merge_mode='concat')
        self.lstm_layer2_z1 = layers.Bidirectional(layers.LSTM(self.z1_rhus[1], return_state=True, time_major=False), merge_mode='concat')

        # fully connected layers for computation of mu and sigma
        self.z1mu_fclayer = layers.Dense(
            z1_dim, activation=mu_nl, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros')

        self.z1logvar_fclayer = layers.Dense(
            z1_dim, activation=logvar_nl, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros')

        self.z2mu_fclayer = layers.Dense(
            z2_dim, activation=mu_nl, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros')

        self.z2logvar_fclayer = layers.Dense(
            z2_dim, activation=logvar_nl, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros')


    def call(self, inputs):
        ''' The full decoder '''
        z2_pre_out = self.z2_pre_encoder(inputs)

        z2_mu = self.z2mu_fclayer(z2_pre_out)
        z2_logvar = self.z2logvar_fclayer(z2_pre_out)
        z2_sample = reparameterize(z2_mu, z2_logvar)

        z1_pre_out = self.z1_pre_encoder(inputs, z2_sample)

        z1_mu = self.z1mu_fclayer(z1_pre_out)
        z1_logvar = self.z1logvar_fclayer(z1_pre_out)
        z1_sample = reparameterize(z1_mu, z1_logvar)

        qz1_x = [z1_mu, z1_logvar]
        qz2_x = [z2_mu, z2_logvar]

        return z1_mu, z1_logvar, z1_sample, z2_mu, z2_logvar, z2_sample, qz1_x, qz2_x

    def z2_pre_encoder(self, x):
        """
        Pre-stochastic layer encoder for z2 (latent sequence variable)
        Args:
            x(tf.Tensor): tensor of shape (bs, T, F)
        Return:
            out(tf.Tensor): concatenation of hidden states of all LSTM layers
        """

        ## Bidirectional LSTM
        outputs1, forward_h1, forward_c1, backward_h1, backward_c1 = self.lstm_layer1_z2(inputs=x, training=True)
        outputs2, forward_h2, forward_c2, backward_h2, backward_c2 = self.lstm_layer2_z2(inputs=outputs1, training=True)
        out = tf.concat([forward_h1, backward_h1, forward_h2, backward_h2], axis=-1)

        return out

    def z1_pre_encoder(self, x, z2):
        """
        Pre-stochastic layer encoder for z1 (latent segment variable)
        Args:
            x(tf.Tensor): tensor of shape (bs, T, F)
            z2(tf.Tensor): tensor of shape (bs, D1)
        Return:
            out(tf.Tensor): concatenation of hidden states of all LSTM layers
        """

        bs, T = tf.shape(input=x)[0], tf.shape(input=x)[1]
        z2 = tf.tile(tf.expand_dims(z2, 1), (1, T, 1))
        x_z2 = tf.concat([x, z2], axis=-1)

        ## Bidirectional LSTMs
        outputs1, forward_h1, forward_c1, backward_h1, backward_c1 = self.lstm_layer1_z1(inputs=x_z2, training=True)
        outputs2, forward_h2, forward_c2, backward_h2, backward_c2 = self.lstm_layer2_z1(inputs=outputs1, training=True)
        out = tf.concat([forward_h1, backward_h1, forward_h2, backward_h2], axis=-1)

        return out


class Decoder(layers.Layer):
    ''' decodes factors z1 and z2 to reconstructed input x'''

    def __init__(self, x_rhus=[256,256], tr_shape=(20, 80),
                 mu_nl=None, logvar_nl=None, name="decoder", **kwargs):

        super(Decoder, self).__init__(name=name,   **kwargs)

        # x
        self.tr_shape = tr_shape

        # RNN specs
        self.x_rhus = x_rhus

        ## Bidirectional LSTMs
        self.lstm_layer1_x = layers.Bidirectional(layers.LSTM(self.x_rhus[0], return_sequences=True, time_major=False), merge_mode='concat')
        self.lstm_layer2_x = layers.Bidirectional(layers.LSTM(self.x_rhus[1], return_sequences=True, time_major=False), merge_mode='concat')

        # fully connected layers for computing mu and logvar
        self.xmu_fclayer = layers.Dense(
            tr_shape[1], activation=mu_nl, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros')

        self.xlogvar_fclayer = layers.Dense(
            tr_shape[1], activation=logvar_nl, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros')


    def call(self, x, z1, z2):

        bs, T = tf.shape(input=x)[0], tf.shape(input=x)[1]
        T_int = x.get_shape().as_list()[1]  # get the integer value directly, to use in for-loop


        z1 = tf.tile(tf.expand_dims(z1, 1), (1, T, 1))
        z2 = tf.tile(tf.expand_dims(z2, 1), (1, T, 1))
        z1_z2 = tf.concat([z1, z2], axis=-1)

        # Uni or bidirectional LSTMs (same calls here)
        outputs1 = self.lstm_layer1_x(inputs=z1_z2, training=True)
        output = self.lstm_layer2_x(inputs=outputs1, training=True)

        x_mu, x_logvar, x_sample = [], [], []
        for timestep in range(0, int(T_int)):
            out_t = output[:, timestep, :]
            x_mu_t = self.xmu_fclayer(out_t)
            x_logvar_t = self.xlogvar_fclayer(out_t)
            x_sample_t = reparameterize(x_logvar_t, x_mu_t)

            x_mu.append(x_mu_t)
            x_logvar.append(x_logvar_t)
            x_sample.append(x_sample_t)

        x_mu = tf.stack(x_mu, axis=1)
        x_logvar = tf.stack(x_logvar, axis=1)
        x_sample = tf.stack(x_sample, axis=1)
        px_z = [x_mu, x_logvar]

        return output, x_mu, x_logvar, x_sample, px_z

class Regulariser(layers.Layer):
    ''' predicts the regularizing factors '''

    def __init__(self, z1_nlabs={}, z2_nlabs={}, name="regulariser", **kwargs):

        super(Regulariser, self).__init__(name=name, **kwargs)

        #dict with e.g. [('gender',3), ('region',9)]
        self.z1_nlabs = z1_nlabs
        self.z2_nlabs = z2_nlabs

        # subtract -1 for the unknown labels (unsupervised),
        # these will get zero logits (see fix_logits below) and do not need to be mapped to by the FC layer
        self.z1_nlabs_per_fac = [nlab - 1 for nlab in list(self.z1_nlabs.values())]
        self.z2_nlabs_per_fac = [nlab - 1 for nlab in list(self.z2_nlabs.values())]

        # fully connected layers for every label
        self.reg_z1_fclayer = layers.Dense(
            sum(self.z1_nlabs_per_fac), activation=None, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros')

        self.reg_z2_fclayer = layers.Dense(
            sum(self.z2_nlabs_per_fac), activation=None, use_bias=True,
            kernel_initializer='glorot_uniform', bias_initializer='zeros')

    def call(self, z1_mu, z2_mu):

        z1_all_rlogits = self.reg_z1_fclayer(z1_mu)
        z2_all_rlogits = self.reg_z2_fclayer(z2_mu)

        z1_rlogits = self.fix_logits(z1_all_rlogits, self.z1_nlabs_per_fac)
        z2_rlogits = self.fix_logits(z2_all_rlogits, self.z2_nlabs_per_fac)

        return z1_rlogits, z2_rlogits

    def fix_logits(self, all_rlogits, nlabs_per_fac):
        # split the logits for z1/z2 into the different factors
        # e.g. rlogits = list(Tens(logits_gender), Tens(logits_region))
        rlogits = []

        for tens in tf.split(all_rlogits, nlabs_per_fac, 1):
            T = tf.shape(input=tens)[0]
            z = tf.zeros([T, 1], dtype=tf.float32)
            # add column of zeros at start for data with unknown labels ('' label always added at start)
            rlogits.append(tf.concat((z, tens), axis=1))
            # rlogits.append(tens)

        return rlogits

#@tf.function
def log_normal_pdf(x, mu=0., logvar=0., raxis=1):
  log2pi = tf.math.log(2. * np.pi)
  return tf.reduce_sum(
          -.5 * ((x - mu) ** 2. * tf.exp(-logvar) + logvar + log2pi),
          axis=raxis)

#@tf.function
def reparameterize(mu, logvar):
    eps = tf.random.normal(shape=mu.shape)
    return eps * tf.exp(logvar * .5) + mu

#@tf.function
def kld(p_mu, p_logvar, q_mu, q_logvar):
    """
    compute D_KL(p || q) of two Gaussians
    """
    # Added extra brackets after the minus sign
    return -0.5 * (1 + p_logvar - q_logvar - (((p_mu - q_mu) ** 2 + tf.exp(p_logvar)) / tf.exp(q_logvar)))
