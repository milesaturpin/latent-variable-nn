# Hierarchical Latent Variable Neural Networks

"Hierarchical" in this context means that we have group-level parameters in addition to our normal global parameters. These group-level parameters are allowed to deviate from another.

A "group" is a collection of data points that are generated from similar processes in a way that we are aware of e.g. letters written by the same person, sentences from the same character in a play, patients from the same hospital. A "group level" is a partitioning of the dataset into one type of group. A dataset can contain more than one group level.

## Prerequisites

1. Create a virtual environment and install packages from `requirements.txt`.

2. Create a directory called `data`. Inside place the directory for each dataset that will be used. Inside that create two separate directories for small and large versions of the dataset. Finally, inside that place the data files partitioned into `x`, `y`, and the group id arrays `gid`, `gid2`.

At the end it should look like this:

```
data/
  femnist/
    large/
      x_train.npy
      gid_train.npy
      y_train.npy
      x_test.npy
      gid_test.npy
      y_test.npy
    small/
      ...
  shakespeare/
    large/
      x_train.npy
      gid_train.npy
      gid2_train.npy
      y_train.npy
      x_test.npy
      gid_test.npy
      gid2_test.npy
      y_test.npy

```

Note that when adding a new dataset, the data doesn't actually have to be partitioned into separate `x`, `y`, `gid`, train-test split etc. It just has to be in the right format by the end of the `read_data` function for the dataset. The head directory `data` can be changed with the command line argument `--data-dir`.

## Running experiments

Example command: `python train.py --lr=0.001 --batch-size=200 --num-epochs=5 --data-dir='data' --dataset='femnist' --data-size='large' --model-size='large' --latent-config='double' --z-dim 10 10 --eval-every=1 --print-freq=10 --seed=27436 --testing --description="added second latent var"`

* `--data-dir` : where we will look for data

* `--dataset` : the name of the specific dataset directory

* `--data-size` : whether to use a large or small version of the data

* `--model-size` : directs code to initialize `_build_model` with a specific set of parameters

* `--latent-config` : this value is used in `main` to pick which model gets used

* `--z-dim` : this one is a little tricky: if you have more than one group or more than one latent variable, put the dimensions of the latent variables separated by spaces. Don't insert an `=`.

* `--eval-every` : how many epochs in between evaluating on test set

* `--print-freq` : how many times progress is printed per epoch

* `--seed` : please set a seed for sanity's sake

* `--testing` : will redirect output to `experiments/tests` to keep main experiment directory clean

* `--description` : longer form note describing context of experiment. This is currently how I'm remembering what each experiment was for which isn't the most reliable system

### Output

Results will be written to `experiments/{dataset}/{datetime}/`. In this directory you will find the files:

* `hyperparams.json` — a JSON parsing of the arguments provided in the command line.

* `model_weights.pickle` — a pickled version of result of calling `model.get_weights()`, which is a built-in keras method that returns a list of arrays. Models can be restored by creating an instance of the relevant model class and calling `model.set_weights(model_weights)`. There's a little bit of a gotcha: the model instance must be called on a piece of data before setting the weights so that Keras can initialize the weights.

* `train.log` — history of train losses and test accuracies that were printed to the terminal.

* `training_stats1.csv`, `training_stats2.csv`, etc. — each file is a csv file where each row contains `epoch, gid, test_acc, test_f1, test_loss`. There's a different file for each epoch. Yes, this is kind of hacky but it's easy enough to concatenate the files before analyzing results.


## Adding a new dataset and model

### 1. Write a `read_data` function in `train.py` for the new dataset

For example, we may write a `read_eICU_data` function which returns two lists of numpy arrays, `(train_data, test_data)`. Each should be of the form `[x, gid, gid2, ..., y]`. This format allows decomposition into `inputs, labels = arr[:-1], arr[-1]`.

Currently, we assume that the train/test split is by user, in other words, the test set consists of subsets of data from users we saw in the training set.

`y` should be given as a a vector of shape `(n,)` i.e. labels are not one-hot encoded.

Information on the grouping structure of the data, the "group ids", is given in arrays of shape `(n,)` titled `gid`, `gid2`, `gid3` etc. with `gid` being the lowest in the hierarchy, or 'closest' to the data. This ordering is not very important, except that group level training statistics (e.g. test accuracy broken down by group) are only provided on the `gid` level, which in the code is equivalent to whatever is in the second entry in `[x, gid, gid2, ..., y]`. Each entry `i` of the group arrays will be some number from `0` to `k-1`, `k` being the number of groups in this group level. Group information should be mapped to id's either in the `read_data` function or directly stored in the input data. The group ids are used to efficiently index into a data structure storing the group-level latent variables, so this format is necessary. To be consistent, make all group id arrays of type `np.int32`.

Once the function is ready, add a predicate to the conditional in `train.py` for when `args.dataset == <new dataset>`. In the conditional, call `read_data` for the new dataset and create the relevant model by creating the relevant model directed by the output of `--latent-config`.

### 2. Create a subclassed model of `BaseModel`

The base model contains documented methods for creating batch iterators, running the training loop, logging training performance, and saving weights. Subclasses are responsible for building the latent variable structure, the model architecture, and the forward pass through the model.

I find having a small and large version of the model is helpful for debugging — I can run the small model on the small version of the dataset when I want to see training run to completion. In the subclassed version of `_build_model` there is logic to pick a set of model hyperparameters depending on the value of `args.model_size`.

Currently the loss and optimizer is defined at the top of `main` in `train.py`.

Other methods:

* `_build_latent_space` — Initialize latent variables and prior. If the latent variable is simply a vector with a normal prior, can use the `latent_normal_vector` function in `models/model_utils.py`.

* `construct_variational_posterior` — given a batch of group ids, use these to construct a batch of variational distributions for each entry in the input. I have been using the Tensorflow Probability/Distributions API to create these distributions. Tensorflow is able to take gradients of distributions parameterized with `tf.Variable` types w.r.t. the parameters of the distribution. They currently don't have this implemented for differentiable Categorical distributions, disallowing the use of mixture models as variational distributions. If the posterior is mean-field normal, can use `latent_vector_variational_posterior` defined in `models/model_utils.py`.

* `call` — given `x`, `gid`, `gid2`, etc. specify the forward pass through the model. In this method, you will invoke `z = construct_variational_posterior(gid).sample()`, and then use `z` somewhere in the model. I currently have some control flow in `call` to make it easy to turn off the latent variables by setting `--latent-config='none'`.

* `call_sample_z` — working on a way to elegantly merge this with the normal call method.

## Model Descriptions

* `NormalCNN` —

* `LatentFactorCNN` —

* `DoubleLatentCNN` —

* `LatentBiasCNN` —

* `NormalLSTM` —

* `LatentFactorLSTM` —

* `DoubleLatentLSTM` —


###  Experimenting with new latent configurations

Currently, if I want to try a new latent variable configuration, I create a new class for it.

The functions `latent_normal_vector` and `latent_vector_variational_posterior` defined in `models/model_utils.py` make it easy to define new latent vectors with normal means and mean-field normal posteriors.

Make sure to import the new model to `train.py`, add an option in `--latent-config` and then use that option to query `model_dict` to get the right model.


## Additional Notes

### Other files

#### `analyze_results.ipynb`

At the time of writing this notebook is still a bit messy. It contains code for pulling in results from the experiments directory and analyzing the latent space, drawing samples, etc.

### Workflow

In experiments I added a directory called `sota`, read "state of the art", to make it easier to find the best experiments.

GPUs significantly speed up training for the larger versions of the models and datasets. I use the `tf.data` API for serving batches as it makes prefetching batches really easy, e.g. `dataset.prefetch(4)`. Depending on how expensive the preprocessing is, prefetching is essential for getting good speed ups. Experimenting with batch size is also a good idea. I use `watch -n0.1 nvidia-smi` to monitor GPU utilization and I try to make sure it's *at least* 50%, and `top -i` to monitor CPUs helping with processing batches.


## Future Work

1. Experiment with different types of latent variables

	* Latent filters for CNN

	* Initial hidden state for RNN

	* Appended to the hidden state during every iteration of RNN

	* Combinations of the above

2. Use amortized inference to compute latent variables for groups.

3. Explore data efficiency of predicting latent variable for new user. Either using standard optimization procedure or amortized inference.

4. Explore self-normalized importance sampling.
