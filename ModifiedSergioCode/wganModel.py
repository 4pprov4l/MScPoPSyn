import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd

from keras.activations import relu, softmax
from keras.callbacks import LearningRateScheduler, EarlyStopping, ReduceLROnPlateau, TerminateOnNaN
from keras.layers import Activation, BatchNormalization, Concatenate, concatenate, Dense, Dropout, Input, InputLayer, Lambda, LeakyReLU
#from keras.layers.merge import _Merge
from keras.losses import mse, binary_crossentropy, categorical_crossentropy, mean_squared_error
import keras.metrics as metrics
from keras.models import Model, Sequential
from keras.utils import plot_model
from keras import backend as K
from keras import metrics

import TUutils

class WGAN(): # Training partly based on https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
    
    def __init__(self, train, validation, numerical_col_n, categorical_col_n, categories_n, categories_cum, eval_set,# Data
                     col_names, original_df, pre_one_hot_df, # These are important to come back from the VAE samples to the original dataset
                     intermediate_dim_gen=256, latent_dim=100, n_hidden_layers_gen=4, # Architecture Generator
                     intermediate_dim_crit=128, n_hidden_layers_crit=2, drop_rate_g=0., drop_rate_c=0.25,# Architecture Critic
                     batch_size=64, epochs=50, gen_learn_rate=0.001, crit_learn_rate=0.00005, # Training
                     clip_value=0.1, nCritic=5): # Training
        
        # Data parameters
        self.data_train = train
        self.data_validation = validation
        
        self.numerical_col_n = numerical_col_n # Scalar
        self.categorical_col_n = categorical_col_n # Scalar
        self.categories_n = categories_n # List of scalars
        self.categories_cum = categories_cum # List of scalars
        self.eval_set = eval_set # Set of variables with which the model will be evaluated
        
        self.col_names = col_names # column names of the one hot encoded dataset
        self.original_df = original_df # original data set, to retrieve its structure
        self.pre_one_hot_df = pre_one_hot_df # one hot encoded dataset, to retrieve its structure
        
        # Architecture parameters
        self.input_dim_crit = validation.shape[1]
        self.latent_dim = latent_dim # Input dimension for generator
        self.intermediate_dim_crit = intermediate_dim_crit
        self.intermediate_dim_gen = intermediate_dim_gen
        self.n_hidden_layers_crit = n_hidden_layers_crit
        self.n_hidden_layers_gen = n_hidden_layers_gen
        self.drop_rate_g = drop_rate_g
        self.drop_rate_c = drop_rate_c
        
        # Training parameters
        self.epochs = epochs
        self.batch_size = batch_size
        self.gen_learn_rate = gen_learn_rate
        self.crit_learn_rate = crit_learn_rate
        self.clip_value = clip_value # WGAN parameter
        self.nCritic = nCritic # WGAN parameter
        self.crit_opt = keras.optimizers.RMSprop(lr=self.crit_learn_rate) # RMSprop
        self.gen_opt = keras.optimizers.RMSprop(lr=self.gen_learn_rate) # RMSprop
        
        # Sampling parameters
        self.n_samples = validation.shape[0]
        
        # Model creation
        self.create_wgan()
        

    # Generator architecture
    def create_generator(self):
        '''
        This is the generator architecture, 

        params: 
        n_hidden_layers_gen: number of hidden layers
        intermediate_dim_gen: value of the number of neurons for the first intermediate layer. 
            They increase in a factor of 2 (N*2). 
        latent_dim: dimension of latent space taken as input for the generator
        
        output:
        The function outputs the generator model in keras. The architecture is defined by the user when
        the whole GAN class is initialized.
        '''
            
        gen_input = Input(shape=(self.latent_dim,), name='gen_input')
        
        # Make the generator intermediate dimensions as it should be first
        self.intermediate_dim_gen = int(self.intermediate_dim_gen/2**(self.n_hidden_layers_gen-1))
        
        # Intermediate layers
        for _ in range(self.n_hidden_layers_gen):
            if _==0: # The first one takes the inputs as input
                #intermediate = Dense(self.intermediate_dim_gen, name= 'generator_hidden_{}'.format(_), kernel_initializer='he_uniform')(gen_input)
                #intermediate = BatchNormalization()(intermediate)
                #intermediate = Activation('relu')(intermediate)
                intermediate = tf.keras.layers.Dense(self.intermediate_dim_gen, name='generator_hidden_{}'.format(_), kernel_initializer='he_uniform')(gen_input)
                intermediate = tf.keras.layers.Activation('relu')(intermediate)
                #intermediate  = LeakyReLU(alpha=0.1)(intermediate)
                intermediate = Dropout(rate=self.drop_rate_g)(intermediate)
            else: # After the first one, the network takes the intermediate layers as input
                intermediate = Dense(self.intermediate_dim_gen, name= 'generator_hidden_{}'.format(_), kernel_initializer='he_uniform')(intermediate)
                #intermediate = BatchNormalization()(intermediate)
                intermediate = Activation('relu')(intermediate)
                #intermediate  = LeakyReLU(alpha=0.1)(intermediate)
                intermediate = Dropout(rate=self.drop_rate_g)(intermediate)
            self.intermediate_dim_gen *= 2 # Update the value of the number of neurons

        # Final layer
        # Categorical decode
        x_decoded_mean_cat = [Dense(self.categories_n[cat], activation='softmax')(intermediate) 
                              for cat in range(len(self.categories_n))]

        if self.numerical_col_n > 0: # If there are numerical variables, concatenate both
            x_decoded_mean_num = Dense(self.numerical_col_n)(intermediate) # Numerical decode
            gen_output = concatenate([x_decoded_mean_num] + x_decoded_mean_cat, name='gen_output')
        else: # If there are no numerical variables only include the cdrop_rate_g=0.,ategorical output layer
            gen_output = concatenate(x_decoded_mean_cat, name='gen_output')

        return Model(inputs=gen_input, outputs=gen_output)
    
    # Critic architecture
    def create_critic(self):
        '''
        COMMENT THE CODE
        '''

        crit_input = Input(shape=(self.input_dim_crit,), name='crit_input')
        
        # Intermediate layers
        for _ in range(self.n_hidden_layers_crit):
            if _==0: # The first one takes the inputs as input
                #intermediate = Dense(self.intermediate_dim_crit, name= 'critic_hidden_{}'.format(_), kernel_initializer='he_uniform')(crit_input)
                #intermediate = BatchNormalization()(intermediate)
                intermediate = tf.keras.layers.Dense(self.intermediate_dim_gen, name='generator_hidden_{}'.format(_), kernel_initializer='he_uniform')(crit_input)
                intermediate = tf.keras.layers.Activation('relu')(intermediate)
                #intermediate = Activation('relu')(intermediate)
                #intermediate  = LeakyReLU(alpha=0.1)(intermediate)
                intermediate = Dropout(rate=self.drop_rate_c)(intermediate)
            else: # After the first one, the network takes the intermediate layers as input
                intermediate = Dense(self.intermediate_dim_crit, name= 'critic_hidden_{}'.format(_), kernel_initializer='he_uniform')(intermediate)
                #intermediate = BatchNormalization()(intermediate)
                intermediate = Activation('relu')(intermediate)
                #intermediate  = LeakyReLU(alpha=0.1)(intermediate)
                intermediate = Dropout(rate=self.drop_rate_c)(intermediate)
            self.intermediate_dim_crit = int(self.intermediate_dim_crit/2) # Update the value of the number of neurons
        
        #intermediate = Dropout(rate=0.1)(intermediate)
        crit_output = Dense(1, name='crit_output')(intermediate)
        
        return Model(crit_input, crit_output)
    
    
    # GAN creation
    def create_wgan(self):
        '''
        COMMENT THE CODE
        '''
        
        # Create and compile the discriminator
        self.critic = self.create_critic()
        self.critic.compile(loss=self.wasserstein_loss, optimizer=self.crit_opt, metrics=['binary_accuracy'])

        # Create the generator
        self.generator = self.create_generator()

        # The generator takes noise as input and generates observations
        z = Input(shape=(self.latent_dim,))
        generated = self.generator(z)

        # For the combined model we will only train the generator
        self.critic.trainable = False

        # The critic takes generated observations as input and critiques
        crit_guess = self.critic(generated)

        # The combined model  (stacked generator and critic)
        # Trains the generator to fool the critic
        self.wgan = Model(z, crit_guess)
        self.wgan.compile(loss=self.wasserstein_loss, optimizer=self.gen_opt) 

        
    #def wasserstein_loss(self, y_true, y_pred):
        '''
        Wasserstein loss. Since we have labels -1 and 1, their product is the expected value of th
        '''
        #return K.mean(y_true * y_pred)

    def wasserstein_loss(self, y_true, y_pred):
        return tf.reduce_mean(y_true * y_pred)

    
    def wgan_fit(self):
        '''
        COMMENT THE CODE
        '''

        # Adversarial ground truths
        valid = -np.ones((self.batch_size, 1))
        fake  =  np.ones((self.batch_size, 1))
        
        # Loss and accuracy lists for graphing purposes
        self.gen_loss  = []
        self.crit_loss = []
        self.crit_acc  = []
        
        for epoch in range(self.epochs):
            
            # ---------------------
            #  Train Discriminator
            # ---------------------

            for _ in range(self.nCritic):
                # Select a random batch of observations
                idx = np.random.choice(self.data_train.shape[0], self.batch_size, replace=False)
                obs = self.data_train[idx]

                noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

                # Generate a batch of new images
                gen_obs = self.generator.predict(noise)

                # Train the discriminator
                crit_loss_real = self.critic.train_on_batch(obs, valid)
                crit_loss_fake = self.critic.train_on_batch(gen_obs, fake)
                
                # Clip the weights
                for l in self.critic.layers:
                    weights = l.get_weights()
                    weights = [np.clip(w, -self.clip_value, self.clip_value) for w in weights]
                    l.set_weights(weights)
                    
            self.crit_loss.append(0.5 * np.add(crit_loss_real[0], crit_loss_fake[0]))
            self.crit_acc.append(0.5 * np.add(crit_loss_real[1], crit_loss_fake[1]))

            # ---------------------
            #  Train Generator
            # ---------------------

            noise = np.random.normal(0, 1, (self.batch_size, self.latent_dim))

            # Train the generator (to have the discriminator label samples as valid)
            self.gen_loss.append(self.wgan.train_on_batch(noise, valid))
            
            if epoch%100==0:
                # Plot the progress
                print ("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, self.crit_loss[-1], 100*self.crit_acc[-1], self.gen_loss[-1]))
                #print (crit_loss_real[0], crit_loss_fake[0], gen_loss)
                #print(gen_obs[1,:], np.sum(gen_obs, axis=1))


    # Sampling helper function for evaluation
    def sampler(self):
        z_sample = np.random.normal(0., 1.0, size=(self.n_samples, self.latent_dim))
        prediction = self.generator.predict(z_sample).transpose()
        samples = np.zeros((self.input_dim_crit, self.n_samples))
        samples[:self.numerical_col_n,:]=prediction[:self.numerical_col_n,:]
        for idx in range(len(self.categories_cum)-1):
            idx_i = self.numerical_col_n+self.categories_cum[idx] # Initial index
            idx_f = self.numerical_col_n+self.categories_cum[idx+1] # Final index
            mask = np.argmax(prediction[idx_i:idx_f, :], axis=0) + idx_i
            samples[mask, np.arange(len(mask))] = 1
                
        return samples
    
    # VAE evaluation
    def wgan_evaluate(self, used_metric='MAE'):
        '''
        COMMENT THE CODE
        '''
        # Fit the model
        self.wgan_fit()
        
        # Evaluate it
        self.samples = self.sampler()
        self.wgan_df = TUutils.samples_to_df(self.samples, col_names=self.col_names, original_df=self.original_df, pre_one_hot_df=self.pre_one_hot_df)
        self.validation_df  = TUutils.samples_to_df(self.data_validation.transpose(), col_names=self.col_names, original_df=self.original_df, pre_one_hot_df=self.pre_one_hot_df)

        ##### Count creator
        self.wgan_df['count'] = 1
        self.wgan_df = self.wgan_df.groupby(self.eval_set, observed=True).count()
        self.wgan_df /= self.wgan_df['count'].sum()

        self.validation_df['count'] = 1
        self.validation_df = self.validation_df.groupby(self.eval_set, observed=True).count()
        self.validation_df /= self.validation_df['count'].sum()

        ##### Merge and difference
        real_and_sampled = pd.merge(self.validation_df, self.wgan_df, suffixes=['_real', '_sampled'], on=self.eval_set, how='outer') # on= all variables
        real_and_sampled = real_and_sampled[['count_real', 'count_sampled']].fillna(0)
        real_and_sampled['diff'] = real_and_sampled.count_real-real_and_sampled.count_sampled
        diff = np.array(real_and_sampled['diff'])
        
        metrics = {}
        metrics['MAE']   = np.mean(abs(diff))
        metrics['MSE']   = np.mean(diff**2)
        metrics['RMSE']  = np.sqrt(np.mean(diff**2))
        metrics['SRMSE'] = metrics['RMSE']/real_and_sampled['count_real'].mean()
        print('Evaluating with {}'.format(used_metric))
        print('Using variables {}'.format(self.eval_set))
        print('MAE:{}, MSE:{}, RMSE:{}, SRMSE:{}'.format(metrics['MAE'], metrics['MSE'], metrics['RMSE'], metrics['SRMSE']))
        
        return metrics[used_metric]