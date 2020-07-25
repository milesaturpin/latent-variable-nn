import numpy as np
import pickle
import time
import os
from tqdm import tqdm, trange
import ipdb

from sklearn.metrics import accuracy_score, f1_score

import tensorflow as tf
import tensorflow_probability as tfp

from utils import robust_loss, round_nums

import re

tfd = tfp.distributions


class BaseModel(tf.keras.Model):
    """
    Base model that serves as super class for different architectures.
    Subclassed models are responsible for building out the latent space
    as well as the model architecture. The base model contains training
    loop code and utilities for saving results.

    Args:
        optimizer : keras optimizer object
        loss_fn : keras loss function object
        num_groups : array-like obj with number of unique groups in
            each level of grouping; used to initialize latent variables
        experiment_dir : used for saving training stats and model weights
        logger : logging object
    """

    def __init__(self, optimizer, loss_fn, train_size, num_groups, group_train_sizes, kl_weight, args, experiment_dir, logger):
        super(BaseModel, self).__init__()
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.experiment_dir = experiment_dir
        tb_file = os.path.join(experiment_dir, 'tensorboard')
        self.train_summary_writer = tf.summary.create_file_writer(tb_file)
        self.logger = logger
        self.model_size = args.model_size
        self.z_dim = args.z_dim
        self.train_size = train_size
        self.num_groups = num_groups
        self.group_train_sizes = group_train_sizes
        self.seed = args.seed
        self.kl_weight = kl_weight
        self._build_model()
        if args.latent_config != 'none':
            self._build_latent_space()


    def _build_model(self):
        """Initialize model layers."""
        pass

    def _build_latent_space(self):
        """Initialize latent variables and their prior distributions."""
        pass


    def create_batch_generator(
        self, data, batch_size, prefetch=2):
        """Use tf.data API to create efficient input pipeline.

        Should be overriden in the subclasses if particular dataset needs
        a different batching procedure.

        Args:
            data : list of arrays of the form [x, gid, gid2, ..., y]
            batch_size : self-explanatory
            prefetch : number of batchs to precompute
        """
        generator = tf.data.Dataset.from_tensor_slices(tuple(data))
        generator = generator.batch(batch_size)
        # Experimental feature, automatically picks # of batches to process
        generator = generator.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
        return generator


    def train(self, train_data, test_data, batch_size, num_epochs, eval_every=1, print_freq=10):
        """Train network.

        Args:
            train_data, test_data : list of numpy arrays in the order
                [x, gid, gid2, ..., y]
            batch_size : self-explanatory
            num_epochs : self-explanatory
            eval_every : number of epochs in between test set evaluations
            print_freq : how many times per epoch to print training
                progress. Useful when individual epochs take a long time
        """
        #self.logger.info('Evaluating untrained model...')
        #self.log_group_test_performance(test_data, epoch=0) 

        ## TODO: find less hacky way to build model, get summary
        #inputs, _ = train_data[:-1], train_data[-1]
        #inputs = [x[:5] for x in inputs]
        #_ = self(*inputs)
        #_ = self.losses
        ## Try to print summary of param counts, won't work for some models
        #try:
        #    self.summary(print_fn=self.logger.info)
        #except:
        #    self.logger.info('Configuration not amenable to `summary`.')


        # Stateful Keras object for keeping track of mean loss
        train_loss = tf.keras.metrics.Mean('train_loss')
        mean_kl_loss = tf.keras.metrics.Mean('kl_loss')

        num_batches = np.ceil(len(train_data[0])/batch_size)
        #self.kl_weight = 1/num_batches

        last_time = time.time()
        for epoch in range(1, num_epochs+1):
            self.logger.info('--- Epoch {} ---'.format(epoch))

            train_generator = self.create_batch_generator(train_data, batch_size)

            for step, batch in enumerate(train_generator):
                
                global_step = (epoch-1)*num_batches + step
                kl_annealing_weight = tf.convert_to_tensor(
                    min(1., (global_step/num_batches) / (num_epochs/2)), dtype=tf.float32)

                loss, kl_loss, grads = self.train_step(
                    batch, 
                    tf.convert_to_tensor(global_step, dtype=tf.int32), 
                    tf.convert_to_tensor(num_batches, dtype=tf.int32), 
                    tf.convert_to_tensor(epoch, dtype=tf.int32), 
                    tf.convert_to_tensor(num_epochs, dtype=tf.int32), 
                    kl_annealing_weight)
                train_loss(loss)
                mean_kl_loss(kl_loss)

                # Print out train loss every 1/print_freq thru train set
                if (step+1) % np.ceil(num_batches/print_freq) == 0 or (step+1) == num_batches:
                    self.logger.info(
                        'Step {} - train loss: {:.5f}, time elapsed: {:d}s'.format(
                            step+1, train_loss.result().numpy(),
                            round(time.time()-last_time)))

                    with self.train_summary_writer.as_default():
                        tf.summary.scalar('loss', train_loss.result(), step=global_step)
                        tf.summary.scalar('kl_loss', mean_kl_loss.result(), step=global_step)

                        names = [
                            re.sub('my_latent_weight_cnn/', '', weight.name)
                            for layer in self.layers
                            for weight in layer.weights]
                        weights = self.get_weights()

                        for name, weight in zip(names, weights):
                            tf.summary.histogram(name+'/weight', weight, step=global_step)

                        for name, grad in zip(names, grads):
                            tf.summary.histogram(name+'/grad', grad, step=global_step)

                        # for i in range(3):
                        #     tf.summary.scalar('dense/w_mu{}'.format(i), self.get_weights()[0][i,0,0], step=step)
                        #     tf.summary.scalar('dense/w_sigma{}'.format(i), self.get_weights()[1][i,0,0], step=step)
                        #     tf.summary.scalar('dense/b_mu{}'.format(i), self.get_weights()[4][i,0], step=step)
                        #     tf.summary.scalar('dense/b_sigma{}'.format(i), self.get_weights()[5][i,0], step=step)
                        # tf.summary.scalar('dense/w0_mu', self.get_weights()[2][0,0], step=step)
                        # tf.summary.scalar('dense/w0_sigma', self.get_weights()[3][0,0], step=step)
                        # tf.summary.scalar('dense/b0_mu', self.get_weights()[6][0], step=step)
                        # tf.summary.scalar('dense/b0_sigma', self.get_weights()[7][0], step=step)

                    last_time = time.time()
                    train_loss.reset_states()
                    mean_kl_loss.reset_states()

            if epoch % eval_every == 0 or epoch == num_epochs:
                self.logger.info('Evaluating test set...')
                self.log_group_test_performance(test_data, epoch=epoch)
                self.save_weights()

    @tf.function
    def train_step(self, batch, global_step, num_batches, epoch, num_epochs, kl_annealing_weight):
        """
        Idiomatic Tensorflow for making predictions and computing
        gradients.
        """
        inputs, labels = batch[:-1], batch[-1]
        with tf.GradientTape() as tape:
            
            #print(inputs[0])
            #print(labels)
            pred = self(*inputs)
            loss = self.loss_fn(labels, pred)
            # Only need to add KL loss once per epoch
            #print( sum(self.losses) / self.train_size)
            #kl_loss = sum(self.losses) / self.train_size
            #import ipdb; ipdb.set_trace()
            

            kl_loss = self.losses
            # if (epoch) % 2 == 0:
            #     kl_annealing_weight = 1
            # else: 
            #     kl_annealing_weight = (global_step/num_batches)-(epoch-1)
            # Esssentially turning annealing off for now
            #kl_annealing_weight = 1.
            #import ipdb; ipdb.set_trace()
            

            # with self.train_summary_writer.as_default():
            #     tf_summ_kl = lambda name, kl: tf.summary.scalar(name, kl, step=global_step)
            #     for i,kll in enumerate(kl_loss):
            #         tf_summ_kl(str(i), kll)
            #     tf_summ_kl('KL annealing weight', kl_annealing_weight)

            
            total_loss = loss + kl_annealing_weight * sum(kl_loss)
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        return loss, kl_loss, grads


    def log_group_test_performance(self, test_data, epoch):
        """Evaluate test set performance across groups.

        Args:
            test_data : list of test arrays
            epoch : current epoch, used for logging results
        """

        # Don't shuffle so that we can match up batch preds with input data
        test_generator = self.create_batch_generator(
            test_data, batch_size=1000)

        scores = []
        # tqdm prints nice progress bars
        for test_batch in tqdm(test_generator):
            inputs, labels = test_batch[:-1], test_batch[-1]
            score = self(*inputs)
            scores.append(score)
        scores = np.concatenate(scores)
        preds = scores.argmax(axis=1)

        output_file = os.path.join(self.experiment_dir,
            'training_stats{}.csv'.format(epoch))
        file = open(output_file, 'w')

        # Header for CSV file, add more columns if add metrics in get_metrics
        file.write('epoch,gid,test_acc,test_f1,test_loss\n')

        # Get stats with respect to first level grouping
        # i.e. whatever group level is in second index
        gid_test = test_data[1]
        y_test = test_data[-1]
        results_list = []
        for gid in trange(self.num_groups[0]):
            # Get instances of each group, calculate performance
            gid_idx = np.where(gid_test == gid)[0]
            gid_metrics = self.get_metrics(y_test[gid_idx], scores[gid_idx])
            results_list.append(gid_metrics)
            # Write out line to csv e.g. `3,347,0.87,0.86,0.34`
            file.write(','.join(map(str, [epoch, gid] + gid_metrics)) + '\n')
        file.close()

        results_arr = np.stack(results_list)
        group_acc = results_arr[:,0]
        overall_acc = accuracy_score(y_test, preds)
        stats = round_nums(
            overall_acc,
            np.percentile(sorted(group_acc), 10),
            np.percentile(sorted(group_acc), 90))

        # TODO: add more metrics
        self.logger.info(
            'Test accuracy: {:.5f}, 10th percentile: {:.5f}, 90th percentile: {:.5f}'.format(*stats))

        with self.train_summary_writer.as_default():
            tf.summary.scalar('test_accuracy', overall_acc, step=epoch)
            tf.summary.histogram('test_loss_group_dist', results_arr[:,2], step=epoch)
            tf.summary.histogram('test_accuracy_group_dist', group_acc, step=epoch)
            tf.summary.histogram('test_f1_group_dist', results_arr[:,1], step=epoch)


    # TODO: add more metrics, if add more need to add to CSV header
    def get_metrics(self, y_true, y_score):
        y_pred = tf.math.argmax(y_score, axis=1)
        return [
            accuracy_score(y_true, y_pred),
            f1_score(y_true, y_pred, average='weighted'),
            robust_loss(y_true, y_score),
        ]


    def save_weights(self):
        """Save weights of model in pickle file."""
        output_file = os.path.join(self.experiment_dir, 'model_weights.pickle')
        with open(output_file, 'wb') as f:
            pickle.dump(self.get_weights(), f)





