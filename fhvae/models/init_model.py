from __future__ import absolute_import, division
import os
import sys
import time
import math
import numpy as np
import tensorflow as tf
from fhvae.models.reg_fhvae_lstm_unidir import RegFHVAE_unidirectional
from fhvae.models.reg_fhvae_lstm_bidir import RegFHVAE_bidirectional
from fhvae.models.reg_fhvae_lstm_atten import RegFHVAE_attention
from fhvae.models.reg_fhvae_transf import RegFHVAEtransf
from fhvae.models.reg_fhvae_lstm import RegFHVAEnew
from fhvae.models.reg_fhvae_lstm_bidir_3layers import RegFHVAE_bidirectional_3layers
from fhvae.models.reg_fhvae_lstm_unidir_3layers import RegFHVAE_unidirectional_3layers
from fhvae.models.reg_fhvae_lstm_bidir_ORIG import RegFHVAE_bidirectional_original

def init_model(conf, finetuning=False):
    ''' initialize model and optimizer'''

    model = create_model(conf)
    optimizer = create_optimizer(conf, finetuning)

    return model, optimizer


def create_model(conf):
    ''' initialize a model '''

    if conf['model'] == 'LSTM_attention':
        model = RegFHVAE_attention(z1_dim=conf['z1_dim'], z2_dim=conf['z2_dim'], z1_rhus=conf['z1_rhus'], z2_rhus=conf['z2_rhus'], x_rhus=conf['x_rhus'], nmu2=conf['nmu2'], z1_nlabs=conf['b_n'], z2_nlabs=conf['c_n'], mu_nl=None, logvar_nl=None, tr_shape=conf['tr_shape'], bs=conf['batch_size'], alpha_dis_z1=conf['alpha_dis_z1'], alpha_dis_z2=conf['alpha_dis_z2'], alpha_reg_b=conf['alpha_reg_b'], alpha_reg_c=conf['alpha_reg_c'], n_phones=conf['num_phones'], bump_logpmu1=conf['bump_logpmu1'], alpha_advreg_b=conf['alpha_advreg_b'], alpha_advreg_c=conf['alpha_advreg_c'], priors=conf['priors'], num_flow_steps=conf['num_flow_steps'], num_noisy_versions=conf.get('num_noisy_versions', 0), alpha_noise=conf.get('alpha_noise', 0.0))

    elif conf['model'] == 'LSTM_unidirectional':
        model = RegFHVAE_unidirectional(z1_dim=conf['z1_dim'], z2_dim=conf['z2_dim'], z1_rhus=conf['z1_rhus'], z2_rhus=conf['z2_rhus'], x_rhus=conf['x_rhus'], nmu2=conf['nmu2'], z1_nlabs=conf['b_n'], z2_nlabs=conf['c_n'], mu_nl=None, logvar_nl=None, tr_shape=conf['tr_shape'], bs=conf['batch_size'], alpha_dis_z1=conf['alpha_dis_z1'], alpha_dis_z2=conf['alpha_dis_z2'], alpha_reg_b=conf['alpha_reg_b'], alpha_reg_c=conf['alpha_reg_c'], n_phones=conf['num_phones'], bump_logpmu1=conf['bump_logpmu1'], alpha_advreg_b=conf['alpha_advreg_b'], alpha_advreg_c=conf['alpha_advreg_c'], priors=conf['priors'], num_flow_steps=conf['num_flow_steps'], num_noisy_versions=conf.get('num_noisy_versions', 0), alpha_noise=conf.get('alpha_noise', 0.0))

    elif conf['model'] == 'LSTM_bidirectional':
        model = RegFHVAE_bidirectional(z1_dim=conf['z1_dim'], z2_dim=conf['z2_dim'], z1_rhus=conf['z1_rhus'], z2_rhus=conf['z2_rhus'], x_rhus=conf['x_rhus'], nmu2=conf['nmu2'], z1_nlabs=conf['b_n'], z2_nlabs=conf['c_n'], mu_nl=None, logvar_nl=None, tr_shape=conf['tr_shape'], bs=conf['batch_size'], alpha_dis_z1=conf['alpha_dis_z1'], alpha_dis_z2=conf['alpha_dis_z2'], alpha_reg_b=conf['alpha_reg_b'], alpha_reg_c=conf['alpha_reg_c'], n_phones=conf['num_phones'], bump_logpmu1=conf['bump_logpmu1'], alpha_advreg_b=conf['alpha_advreg_b'], alpha_advreg_c=conf['alpha_advreg_c'], priors=conf['priors'], num_flow_steps=conf['num_flow_steps'], num_noisy_versions=conf.get('num_noisy_versions', 0), alpha_noise=conf.get('alpha_noise', 0.0))

    elif conf['model'] == 'transformer':
        model = RegFHVAEtransf(z1_dim=conf['z1_dim'], z2_dim=conf['z2_dim'], nmu2=conf['nmu2'], x_rhus=conf['x_rhus'], tr_shape=conf['tr_shape'], z1_nlabs=conf['b_n'], z2_nlabs=conf['c_n'], mu_nl=None, logvar_nl=None, d_model=conf['d_model'], num_enc_layers=conf['num_enc_layers'], num_heads=conf['num_heads'], dff=conf['dff'], pe_max_len=conf['pe_max_len'], rate=conf['rate'], alpha_dis_z1=conf['alpha_dis_z1'], alpha_dis_z2=conf['alpha_dis_z2'], alpha_reg_b=conf['alpha_reg_b'], alpha_reg_c=conf['alpha_reg_c'], n_phones=conf['num_phones'], bump_logpmu1=conf['bump_logpmu1'], alpha_advreg_b=conf['alpha_advreg_b'], alpha_advreg_c=conf['alpha_advreg_c'], priors=conf['priors'], num_flow_steps=conf['num_flow_steps'], num_noisy_versions=conf.get('num_noisy_versions', 0), alpha_noise=conf.get('alpha_noise', 0.0))

    elif conf['model'] == 'LSTM_testadv':
        model = RegFHVAEnew(z1_dim=conf['z1_dim'], z2_dim=conf['z2_dim'], z1_rhus=conf['z1_rhus'], z2_rhus=conf['z2_rhus'], x_rhus=conf['x_rhus'], nmu2=conf['nmu2'], z1_nlabs=conf['b_n'], z2_nlabs=conf['c_n'], mu_nl=None, logvar_nl=None, tr_shape=conf['tr_shape'], bs=conf['batch_size'], alpha_dis_z1=conf['alpha_dis_z1'], alpha_dis_z2=conf['alpha_dis_z2'], alpha_reg_b=conf['alpha_reg_b'], alpha_reg_c=conf['alpha_reg_c'], n_phones=conf['num_phones'], bump_logpmu1=conf['bump_logpmu1'], alpha_advreg_b=conf['alpha_advreg_b'], alpha_advreg_c=conf['alpha_advreg_c'], priors=conf['priors'], num_flow_steps=conf['num_flow_steps'], num_noisy_versions=conf.get('num_noisy_versions', 0), alpha_noise=conf.get('alpha_noise', 0.0))

    elif conf['model'] == 'LSTM_bidirectional_3layers':
        model = RegFHVAE_bidirectional_3layers(z1_dim=conf['z1_dim'], z2_dim=conf['z2_dim'], z1_rhus=conf['z1_rhus'], z2_rhus=conf['z2_rhus'], x_rhus=conf['x_rhus'], nmu2=conf['nmu2'], z1_nlabs=conf['b_n'], z2_nlabs=conf['c_n'], mu_nl=None, logvar_nl=None, tr_shape=conf['tr_shape'], bs=conf['batch_size'], alpha_dis_z1=conf['alpha_dis_z1'], alpha_dis_z2=conf['alpha_dis_z2'], alpha_reg_b=conf['alpha_reg_b'], alpha_reg_c=conf['alpha_reg_c'], n_phones=conf['num_phones'], bump_logpmu1=conf['bump_logpmu1'], alpha_advreg_b=conf['alpha_advreg_b'], alpha_advreg_c=conf['alpha_advreg_c'], priors=conf['priors'], num_flow_steps=conf['num_flow_steps'], num_noisy_versions=conf.get('num_noisy_versions', 0), alpha_noise=conf.get('alpha_noise', 0.0))

    elif conf['model'] == 'LSTM_unidirectional_3layers':
        model = RegFHVAE_unidirectional_3layers(z1_dim=conf['z1_dim'], z2_dim=conf['z2_dim'], z1_rhus=conf['z1_rhus'], z2_rhus=conf['z2_rhus'], x_rhus=conf['x_rhus'], nmu2=conf['nmu2'], z1_nlabs=conf['b_n'], z2_nlabs=conf['c_n'], mu_nl=None, logvar_nl=None, tr_shape=conf['tr_shape'], bs=conf['batch_size'], alpha_dis_z1=conf['alpha_dis_z1'], alpha_dis_z2=conf['alpha_dis_z2'], alpha_reg_b=conf['alpha_reg_b'], alpha_reg_c=conf['alpha_reg_c'], n_phones=conf['num_phones'], bump_logpmu1=conf['bump_logpmu1'], alpha_advreg_b=conf['alpha_advreg_b'], alpha_advreg_c=conf['alpha_advreg_c'], priors=conf['priors'], num_flow_steps=conf['num_flow_steps'], num_noisy_versions=conf.get('num_noisy_versions', 0), alpha_noise=conf.get('alpha_noise', 0.0))

    elif conf['model'] == 'LSTM_bidirectional_original':
        model = RegFHVAE_bidirectional(z1_dim=conf['z1_dim'], z2_dim=conf['z2_dim'], z1_rhus=conf['z1_rhus'], z2_rhus=conf['z2_rhus'], x_rhus=conf['x_rhus'], nmu2=conf['nmu2'], z1_nlabs=conf['b_n'], z2_nlabs=conf['c_n'], mu_nl=None, logvar_nl=None, tr_shape=conf['tr_shape'], bs=conf['batch_size'], alpha_dis_z1=conf['alpha_dis_z1'], alpha_dis_z2=conf['alpha_dis_z2'], alpha_reg_b=conf['alpha_reg_b'], alpha_reg_c=conf['alpha_reg_c'], n_phones=conf['num_phones'], bump_logpmu1=conf['bump_logpmu1'], alpha_advreg_b=conf['alpha_advreg_b'], alpha_advreg_c=conf['alpha_advreg_c'], priors=conf['priors'], num_flow_steps=conf['num_flow_steps'], num_noisy_versions=conf.get('num_noisy_versions', 0), alpha_noise=conf.get('alpha_noise', 0.0))

    else:
        # default
        model = RegFHVAEnew(z1_dim=conf['z1_dim'], z2_dim=conf['z2_dim'], z1_rhus=conf['z1_rhus'], z2_rhus=conf['z2_rhus'], x_rhus=conf['x_rhus'], nmu2=conf['nmu2'], z1_nlabs=conf['b_n'],
  z2_nlabs=conf['c_n'], mu_nl=None, logvar_nl=None, tr_shape=conf['tr_shape'], bs=conf['batch_size'], alpha_dis_z1=conf['alpha_dis_z1'], alpha_dis_z2=conf['alpha_dis_z2'],
  alpha_reg_b=conf['alpha_reg_b'], alpha_reg_c=conf['alpha_reg_c'], n_phones=conf['num_phones'], bump_logpmu1=conf['bump_logpmu1'], alpha_advreg_b=conf['alpha_advreg_b'], alpha_advreg_c=conf['alpha_advreg_c'], priors=conf['priors'], num_flow_steps=conf['num_flow_steps'], num_noisy_versions=conf.get('num_noisy_versions', 0), alpha_noise=conf.get('alpha_noise', 0.0))

    return model


def create_optimizer(conf, finetuning):
    ''' initialize an optimizer '''

    if not finetuning:
        if conf['lr'] == 'custom':
            learning_rate = CustomSchedule(conf['d_model'], warmup_steps=conf['warmup_steps'], k=conf['k'])
        else:
            learning_rate = conf['lr']

        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate, beta_1=conf['beta1'], beta_2=conf['beta2'], epsilon=conf['adam_eps'], amsgrad=True)

        return optimizer

    if finetuning:
        nesterov = conf.get('nesterov', True) in ['True','true','Yes','yes']
        optimizer = tf.keras.optimizers.SGD(learning_rate=conf.get('lr_finetune', 1e-5), momentum=conf.get('momentum', 0.9), nesterov=nesterov)

        return optimizer

class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    # define custom learning rate schedule as according to transformer paper
    def __init__(self, d_model, warmup_steps=7000, k=10):
        super(CustomSchedule, self).__init__()

        self.d_model = tf.cast(d_model, tf.float32)
        self.warmup_steps = tf.cast(warmup_steps, tf.float32)
        self.k = tf.cast(k, tf.float32)

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)

        return self.k * tf.math.rsqrt(self.d_model) * tf.math.minimum(arg1, arg2)
