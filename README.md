# ICU discharge policy - Reinforcement learning

Learning a discharge policy when the ICU is full. This project classify patients based on the predictive mortality rate. The policy decides which class patients to discharge when the ICU is full.

### Installing

Use conda to install the dependecies.

```
conda env create -f environment.yml
```

## Running the scripts
The data file should be renamed to data.xlsx and note that the code expects different header.

To train a NN model, first create a model/ directory and then run

```
python train.py -c <# of class>
```

The model will be saved in model/ directory. To find the optimal policy, first specify the parameters in train_rl.py, then run

```
python train_rl.py
```
