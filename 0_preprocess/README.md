# Data Preprocess

The file structure of the dataset folder should be as follows:

```markdown
.
└── ExtractData/
    ├── BlackEigenvalue/
    │   ├── MatchIDFolder/
    │   │   ├── MatchID_PlayerID.csv
    │   │   ├── ...
    │   │   ├── MatchID_settle.csv
    │   ├── ...
    └── WhiteEigenvalue/
        ├── MatchIDFolder/
        │   ├── MatchID_PlayerID.csv
        │   ├── ...
        │   ├── MatchID_settle.csv
        └── ...
```

We need to extract all the player features from these raw data.
All the feature scripts are in `features` folder, first we use `main.py` to extract players' features.

```sh
$ python main.py BlackEigenvalue
```

The option `dataset_path` is the folder path of your dataset containing MatchIDFolders, for example here we use `BlackEigenvalue` or `WhiteEigenvalue`. Then you'll get all the feature csv files in your MatchIDFolders, and we need to merge all the features.

```sh
$ python feature_merge.py BlackEigenvalue
```
Similar to `main.py`, the option is the folder path of your dataset containing MatchIDFolders. Then we can get battle behavior feature files and match behavior feature files in the root dataset folder, for example here is `ExtractData` folder.

If you run the script for `BlackEigenvalue` and `WhiteEigenvalue` folders, then you can get four csv files showing the player behavior features of cheaters and normal players under different dimensions. As shown below:

```markdown
.
└── ExtractData/
    ├── BlackEigenvalue/
    │   ├── ...
    ├── WhiteEigenvalue/
    │   └── ...
    ├── BlackEigenvalue_battleBehavior.csv
    ├── BlackEigenvalue_matchBehavior.csv
    ├── WhiteEigenvalue_battleBehavior.csv
    └── WhiteEigenvalue_matchBehavior.csv
```
