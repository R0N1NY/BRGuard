# AdaptLink

This repository contains instructions for running the **AdaptLink** subsystem. The process is outlined in the following steps:

## 1. Dataset Preparation

First we need to merge the predictions from the previous subsystems (MatchScope and BattleScan) to build the AdaptLink dataset. You should get the dataset contains the previous subsystems' prediction results in the dataset folder as shown below:

```markdown
.
└── 3_AdaptLink
    ├── mobile/
    │   ├── test/
    │   │   ├── BattleScan_test.csv
    │   │   ├── MatchScope_test.csv
    │   └── val/
    │       ├── BattleScan_val.csv
    │       └── MatchScope_val.csv
    ├── pc/
    │   ├── test/
    │   │   ├── BattleScan_test.csv
    │   │   ├── MatchScope_test.csv
    │   └── val/
    │       ├── BattleScan_val.csv
    │       └── MatchScope_val.csv
    └── other folders and scripts ...
```

Then, we should merge the prediction results to construct a validation set. At the same time, we merge the test set as well, which is convenient for the following testing.

```sh
$ python merge_subsys_pred.py 
```

After running this script, we can get all the dataset we need for training and testing in the `mobile` and `pc` folder.

## 2. Training

We need to find the weights of subsystems and the threshold via the Task-Specified Threshold Optimizer.

```sh
$ python AdaLink.py mobile
```

The option is the platform type (`mobile` or `pc`) of your training target.
After running the script, you can get the values of the weights and thresholds under different metrics, for example:\

```markdown
Optimal Parameters for Accuracy when Recall > 0.85:
Best Accuracy (with Recall > 0.85): 0.9779179810725552
Best Weights: [0.05103119 0.03980747 0.52915914 0.06095147 0.02168069 0.29737004]
Best Threshold: 0.5256941311073032
{
    "TP": 405,
    "FP": 86,
    "TN": 6105,
    "FN": 61,
    "Accuracy": 0.9779179810725552,
    "F1 Score": 0.8463949843260188,
    "Recall": 0.869098712446352,
    "AUC-ROC": 0.9276037900787727
}
```
## 5. Evaluating Predictions

First, enter the `eval` folder and run the `merge_pred.py` script to merge all prediction results. You can choose to merge both validation and test results or just one of them (add the option `test` or `val`):

```sh
$ cd eval && python merge_pred.py mobile
```

The option is the platform type (`mobile` or `pc`) of your evaluation target.
After running the script, you will obtain the final prediction results for the MatchScope subsystem in the specified folder (test/val). Then, run the `evaluation.py` script to obtain the model's evaluation metrics, which will be displayed in the terminal and saved in the respective folder:

```sh
$ python evaluation.py mobile
```

Similar to the merge script, selecting the platform type and you can also choose to evaluate results for a specific dataset, just add the option `test` or `val`.