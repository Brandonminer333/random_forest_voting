import numpy as np
import pandas as pd
from scipy.stats import mode
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


def make_random_dataset(n: int = 1000,
                        p: int = 5,
                        strength: float = 0.8,
                        noise: float = 0.2,
                        bins: int = 2,
                        random_state: int = None) -> tuple[np.array]:
    """
    Create a random dataset with n rows and p columns.
    The last column is a weighted combination of the first p-1 columns
    with some randomness controlled by `strength` and `noise`.

    Parameters
    ----------
    n : int
        Number of rows.
    p : int
        Number of columns (including target column).
    strength : float, optional
        How strongly the last column depends on the others (0–1).
        1.0 means perfect linear combination; 0.0 means pure noise.
    noise : float, optional
        Scale of random noise added to the target.
    random_state : int or None
        Random seed for reproducibility.

    Returns
    -------
    X : numpy.array
        Synthetic features
    Y : numpy.array
        Synthetic predictor based on features
    """
    if n < 1 or int(n) != n:
        raise ValueError(" n must be a positive integer")
    if p < 2:
        raise ValueError('p must be a positive integer greater than 1')

    rng = np.random.default_rng(random_state)

    # Generate independent features
    X = rng.normal(size=(n, p - 1))

    # Generate random weights
    weights = rng.uniform(-1, 1, size=(p - 1))

    # Create target (dependent variable)
    y_signal = X @ weights
    y_noise = rng.normal(scale=noise, size=n)
    y = strength * y_signal + (1 - strength) * y_noise
    y = bin_array(y, bins)

    return train_test_split(X, y, random_state=123)


def bin_array(x: np.array, n_bins: int) -> np.array:
    """
    Transform continuous data to categorical
    Parameters
    ----------
    x: numpy.array
        Continuous data to transform into categorical
    n_bins: int
        Number of categories

    Returns
    -------
    binned : numpy.array
        Transformed x
    """
    # Compute bin edges
    bins = np.linspace(np.min(x), np.max(x), n_bins)
    # Digitize assigns each x to a bin index (1-based)
    binned = np.digitize(x, bins) - 1  # make it 0-based
    return binned


def get_predictions(rf: RandomForestClassifier, X: np.array) -> np.array:
    """
    Get predictions for an input X extrapolated to the ending leaf node 
    of each tree in the forest
    Parameters
    ----------
    rf : sklearn.ensembleRandomForestClassifier.()
        Random Forest Classifier fitted model
    X : nump.array
        Data to predict

    Returns
    -------
    df : 3D numpy.ndarray (n_samples, n_trees, n_classes)
        Array of predictions for all rows/sample of X
            Each prediction is a list of all individual tree predictions in the RandomForestClassifier
                Each tree prediction is a list of samples in the node from training
    """
    # Get the leaf index for each sample in each tree
    leaf_indices = rf.apply(X)

    # Collect the corresponding leaf values
    leaf_values = np.zeros((X.shape[0], len(rf.estimators_), rf.n_classes_))
    for t, tree in enumerate(rf.estimators_):
        tree_values = tree.tree_.value.squeeze(
            axis=1)  # shape (n_nodes, n_classes)
        leaf_values[:, t, :] = tree_values[leaf_indices[:, t]]

    return leaf_values


def accuracy(preds: np.array, Y: np.array, method: str = 'majority'
             ) -> tuple[float, pd.DataFrame]:
    """
    Compute overall classification accuracy and per-class confusion breakdown
    from a set of per-tree predictions (e.g. from a Random Forest).

    Parameters
    ----------
    preds : np.ndarray of shape (n_samples, n_trees, n_classes)
        Model prediction probabilities or scores for each sample, 
        for each tree, across all classes.
        - Axis 0: samples (n_samples)
        - Axis 1: trees (n_trees)
        - Axis 2: class probabilities or logits (n_classes)

    Y : np.ndarray of shape (n_samples,)
        Ground-truth class labels.

    method : str, default='majority'
        Aggregation method to combine tree-level predictions into a 
        final prediction per sample. Supported methods:
        - 'majority' : uses majority voting among tree class predictions.
        - 'weighted' or 'probability' : averages class probabilities across trees, 
          then picks the class with the highest mean probability.
        - 'ranked' : iterative elimination — repeatedly removes the least popular 
          classes among trees until only up to 2 remain, then picks the one with 
          the highest count.

    Returns
    -------
    acc : float
        Overall classification accuracy (mean of correct predictions).

    df : pd.DataFrame
        Per-class confusion breakdown, as returned by `class_confusion_breakdown(Y, final_preds)`.

    Notes
    -----
    - The function assumes that `preds` contains either class probabilities or 
      one-hot predictions per tree.
    - For the 'ranked' method, ties among least frequent classes cause early exit.
    """

    # Convert per-tree probability predictions to class predictions
    # → shape: (n_samples, n_trees)
    # Each element tree_preds[i, j] is the predicted class index for sample i by tree j.
    tree_preds = np.argmax(preds, axis=2)
    n_samples, n_trees = tree_preds.shape

    # --- Aggregate predictions across trees based on chosen method ---
    match method:
        case 'majority':
            # Take the most common class per sample across all trees
            majority_votes = mode(tree_preds, axis=1,
                                  keepdims=False).mode.squeeze()
            final_preds = majority_votes

        case 'weighted' | 'probability':
            # Average predicted class probabilities across trees,
            # then select class with highest mean probability per sample
            mean_probs = preds.mean(axis=1)
            final_preds = np.argmax(mean_probs, axis=1)

        case 'ranked':
            # Iteratively eliminate least common predicted classes per sample
            final_preds = np.zeros(n_samples, dtype=int)
            for i in range(n_samples):
                votes = tree_preds[i, :]
                classes, counts = np.unique(votes, return_counts=True)

                # Repeatedly remove least-voted classes until ≤ 2 remain
                while len(classes) > 2:
                    min_count = counts.min()
                    if np.all(counts == min_count):  # all classes tied
                        break
                    keep_mask = counts > min_count
                    classes, counts = classes[keep_mask], counts[keep_mask]

                # Pick the class with the highest remaining vote count
                winner = classes[np.argmax(counts)]
                final_preds[i] = winner

        case _:
            raise ValueError(f"Unknown method: {method}")

    # --- Compute overall accuracy ---
    acc = np.mean(final_preds == Y)

    # --- Compute confusion breakdown (assumes external helper function) ---
    df = class_confusion_breakdown(Y, final_preds)

    return acc, df


def class_confusion_breakdown(y_true: np.ndarray, y_pred: np.ndarray
                              ) -> pd.DataFrame:
    """
    Compute per-class confusion matrix breakdown: TP, TN, FP, FN for each class.
    """
    classes = np.unique(np.concatenate([y_true, y_pred]))
    data = []
    for cls in classes:
        tp = np.sum((y_pred == cls) & (y_true == cls))
        fp = np.sum((y_pred == cls) & (y_true != cls))
        fn = np.sum((y_pred != cls) & (y_true == cls))
        tn = np.sum((y_pred != cls) & (y_true != cls))
        data.append([cls, tp, tn, fp, fn])

    df = pd.DataFrame(data, columns=["class", "TP", "TN", "FP", "FN"])
    df.set_index("class", inplace=True)
    return df


def class_proportions(y: np.ndarray) -> pd.DataFrame:
    """
    Compute the proportion (and count) of each class label in y.

    Parameters
    ----------
    y : np.ndarray
        Array of class labels.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns:
        - count: number of samples for the class
        - proportion: count / total_samples
    """
    classes, counts = np.unique(y, return_counts=True)
    total = counts.sum()
    proportions = counts / total

    df = pd.DataFrame({
        "class": classes,
        "count": counts,
        "proportion": proportions
    }).set_index("class")

    return df
