Kaggler Loan
============

## Algorithm

1. `alg_1`: Use K-means for clustering and data imputation. Use python's quantile regression for training.
2. `alg_2`: Use features selected by Dboy. Use SVM for classification and regression.

## Convention

- `orig/` will contain original data from Kaggle. We won't commit these files into git.
- `common/` will contain common code (usually in R) used by algorithms.
- `python_code/` will contain python code (maintained by Mark for now).

## How to commit code

- Please create your own branch first:

```
git checkout -b whatever-you-like-for-branch-name master
```

- Make some change and commit it:

```
git add .; git commit -m "Note for this commit"
```

- Open a pull request for code reviewing.
