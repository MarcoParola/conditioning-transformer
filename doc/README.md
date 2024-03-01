# **Documentation**

The project is composed of the following modules, more details are below:

- [Main scripts for training and test models](#main-scripts-for-training-and-test-models)
- [Deep learning architecture implementation](#deep-learning-models)
- [Configuration handling](#configuration-handling)
- [Additional utility scripts](#additional-utility-scripts)


## Main scripts for training and test models

All the experiments consist of train and test transformer architectures. You can use `train.py` and `test.py`, respectively. 

```bash
python train.py
python test.py weight=path/to/model-weight
```


## Deep learning models

## Configuration handling
The configuration managed with [Hydra](https://hydra.cc/). Every aspect of the configuration is located in `config/` folder. The file containing all the configuration is `config.yaml`.

## Additional utility scripts

In the `scripts/` folder, there are all independent files not involved in the `pytorch` workflow for data preparation and visualization.


We can split the dataset with the holdout method.
```bash
python -m scripts.split-dataset --folder data
```

`dataset-stats.py` script can be used to check the number of occurrences per class in a given set of data.
```sh
python -m scripts.dataset-stats --dataset data/Train.json # training set
python -m scripts.dataset-stats --dataset data/Valid.json # validation set
python -m scripts.dataset-stats --dataset data/Test.json # test set
```
