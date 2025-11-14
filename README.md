# Random Forest Voting

## A case-study style analysis of how different tree aggregation methods affect MSE and accuracy.

Project Structure
```bash
.
├── README.md
├── environment.yml
└── code/
    ├── ClassifierVoting.ipynb  # main interactible notebook
    └── ClassifierVoting.py     # Voting utilities for classification
```


`code/ClassifierVoting.ipynb` is the easiest place to toy with the random forests from the synthetic data to the voting method of the forest.



### Installation

This project uses an environment.yml so you can reproduce the exact environment:
```bash
conda env create -f environment.yml
conda activate rf-voting
```

If you prefer `pip`, you can manually install the packages listed inside the YAML.

### Running the Notebook

1. Install the environment

2. Start Jupyter or VScode:

3. Open `code/ClassifierVoting.ipynb`

4. Run cells, tweak parameters, and experiment

### What You Can Do With the Code

The project is meant to be played with. A few ideas:


- Explore how tree constraints change voting behavior

Try adjusting parameters like:

- `max_depth`

- `min_samples_leaf`

- `max_features`

- `bootstrap=False`

Then compare how majority vs weighted voting behave.


### Summary

This project is a practical sandbox for experimenting with:

How Random Forests combine per-tree predictions

How alternative voting methods behave in practice

When (if ever) non-standard aggregation matters

It is not meant to be a deep theoretical treatment, though a my findings are captured [here](https://brandonminer333.github.io/random_forest_voting/).

The code here is intentionally simple, clean, and easy to modify.