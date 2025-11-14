# **A Case Study on the Impacts of Random Forest Prediction Aggregation**

### **Brandon Miner**

## **Introduction**

This discussion focuses on the machine-learning model known as a **Random Forest**, an ensemble technique built from many **decision trees**.
A decision tree is a model that recursively splits the data using a greedy algorithm to form increasingly homogeneous subsets. While decision trees are intuitive and powerful, they tend to **overfit**. This means that they generalize poorly.

Random Forests address this weakness through **ensembling**: they build many decorrelated, shallowly constrained trees (via feature subsampling, bootstrapping, and depth/leaf restrictions). Each individual tree may still overfit, but **aggregating** their predictions reduces variance and improves stability. You can think of it as essentially “averaging away” the overfitting.

But that raises the central question of this post:

> **How exactly are the predictions from individual trees combined?**

In common libraries such as `scikit-learn`, each tree in a classification forest returns the **class probabilities** from its leaf node. Each tree votes for the class with the highest probability, and the forest predicts the class that receives the most votes.

But are other voting methods possible? And do they matter?

## **Voting Methods**

I explored three voting approaches:

1. **Majority Voting**
2. **Weighted Voting**
3. **Ranked Voting** (inspired by real-world voting systems)

### **1. Majority Voting**

This is the most familiar aggregation method.
Each tree predicts a class; the forest outputs the **mode** of these predictions.

It is simple, fast, and fully parallelizable. However, it ignores the **confidence** of each prediction. Two trees with weak evidence count just as much as one tree with very strong evidence.

### **2. Weighted Voting**

Consider three trees and two classes, (A) and (B).

* Tree 1: Leaf contains 4 samples $\to$ 3 are class (A)
* Tree 2: Leaf contains 4 samples $\to$ 3 are class (A)
* Tree 3: Leaf contains 10 samples $\to$ all 10 are class (B)

Majority voting elects class **A**, because two trees predict (A).
But weighted voting notes that Tree 3 has a stronger signal—its leaf has more samples, and those samples unanimously support (B).

In weighted voting, each tree’s vote is weighted by the **class probability** (or equivalently, the proportion of samples in the leaf). Predictions with higher confidence influence the final output more.

This method better captures the actual distribution of the training data within each leaf, making it conceptually closer to probability aggregation.

### **3. Ranked Voting**

This custom method becomes interesting with **three or more classes**.

Example: classes (A), (B), and (C); 100 total trees:

* 45 trees $\to$ (A)
* 35 trees $\to$ (B)
* 20 trees $\to$ (C)

The 20 votes for (C) are nowhere close to winning, but they contain information. In ranked voting, we look at the **second-choice class** for each tree predicting (C). If they overwhelmingly favor (B), then (B) may actually be preferred in a two-class comparison.

For more than three classes, the process repeats: we eliminate the least popular class and redistribute its votes according to the next-ranked option until only two classes remain.

It’s analogous to **instant-runoff voting** in political elections.

## **Findings**

Surprisingly, in most experiments the three methods produced **nearly identical accuracy**, often matching to **three decimal places**. Changing sample size did not meaningfully affect this similarity.

However, when I adjusted parameters like `min_samples_leaf` and `max_depth`, differences began to emerge. Under these conditions, **weighted voting consistently outperformed** majority and ranked voting. Both majority voting and ranked voting declined in similar ways as the trees became more restricted.

This result helps explain why **scikit-learn’s RandomForestClassifier uses weighted aggregation internally**:
it tends to be the most statistically reliable method, especially when tree structures are constrained.

## **Regression**

Random Forest **regression** is different.
Each leaf node returns the **mean** of the target values in that leaf. The forest prediction is simply the **mean of the tree means**.

Because the mean is a linear operator, any alternative aggregation (e.g., weighted mean, median, majority rule) is either:

* **mathematically equivalent** under standard assumptions, or
* **statistically inferior** (e.g., more biased or higher variance)

Thus, unlike classification, regression does not benefit from alternative aggregation strategies.
