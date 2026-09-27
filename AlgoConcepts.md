# Machine Learning Core Algorithms: First Principles & Mathematics

This repository serves as a deep-dive reference for understanding the mathematical intuition and mechanical logic behind core Machine Learning algorithms. The focus is strictly on understanding *how* and *why* the algorithms make decisions from scratch, prioritizing raw mechanics over library implementations.

---
## 1. Linear Regression & Gradient Descent

Linear Regression predicts a continuous numerical output by fitting a straight line (or hyperplane in higher dimensions) through data points. The goal is to find the optimal weights (parameters) that minimize the error between predictions and actual values.

### A. The Hypothesis Function
The hypothesis is the mathematical equation of the line. For a dataset with features $x_1, x_2, ... x_n$, the model calculates the weighted sum of inputs plus a bias term ($\theta_0$):
$$h_\theta(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + ... + \theta_n x_n = \theta^T x$$

### B. The Cost Function (Mean Squared Error)
To measure how "wrong" the line is, we calculate the difference between our prediction $h_\theta(x^{(i)})$ and the actual target $y^{(i)}$, square it to remove negative signs and heavily penalize large errors, and average it across all $m$ data points.
$$J(\theta) = \frac{1}{2m} \sum_{i=1}^{m} (h_\theta(x^{(i)}) - y^{(i)})^2$$

### C. Gradient Descent (GD) Optimization
Gradient Descent calculates the derivative (slope) of the cost function with respect to each weight. The derivative acts as a compass pointing toward the steepest ascent; we subtract it to move "downhill" toward the minimum error. 
$$\theta_j := \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta)$$
*(Where $\alpha$ is the learning rate, controlling the size of the step).*

*   **Batch Gradient Descent (BGD):** Calculates the error across the *entire* dataset before taking a single step. It takes a perfect, smooth path to the minimum but is computationally expensive for large datasets.
*   **Stochastic Gradient Descent (SGD):** Calculates the error and updates the weights using only *one single data point* at a time. It wanders erratically but learns incredibly fast and escapes local minimums easily.

---


## 2. Logistic Regression

Unlike Linear Regression, which predicts a continuous numerical output, Logistic Regression is designed specifically for classification by predicting probabilities between 0 and 1.

*   **The Transformation:** It takes a standard linear equation ($y = mx + b$) and wraps it in a **Sigmoid function**. This squashes any continuous number into an S-curve that never dips below 0 or goes above 1.
*   **The Cost Function (Log Loss):** Because the Sigmoid curve is non-linear, using standard Mean Squared Error (MSE) creates a wavy, non-convex terrain that traps Gradient Descent. Instead, Logistic Regression uses **Log-Likelihood (Cross-Entropy or Log Loss)**.
*   **Intuition:** Log Loss heavily penalizes a model for being *confidently wrong*. If the model predicts a 0.99 probability for a class that is actually 0, the logarithmic penalty skyrockets to near-infinity.

$$LogLoss = - \frac{1}{N} \sum_{i=1}^{N} [y_i \log(\hat{y}_i) + (1-y_i)\log(1-\hat{y}_i)]$$

---

## 3. K-Nearest Neighbors (KNN)

KNN is a "lazy learner." It has no true training phase; it simply memorizes the entire dataset and classifies unseen data based on physical proximity in geometric space.

*   **The Math:** It calculates the distance between a new point and all existing points using **Euclidean Distance** (the Pythagorean theorem expanded into multiple dimensions).

$$d = \sqrt{\sum_{i=1}^{n} (q_i - p_i)^2}$$

*   **The Curse of Dimensionality:** KNN breaks down as you add more features (columns). Every new column adds a spatial dimension. In high-dimensional space, the mathematical volume expands exponentially, making everything incredibly far apart. The concept of "nearest" loses its geometric meaning, severely degrading accuracy.

---

## 4. Decision Trees

Decision Trees abandon continuous equations and geometric distances entirely. Instead, they use a recursive series of rigid, logical True/False questions to slice the dataset into perfectly pure segments. 

Because True/False splits are step functions, they have no derivatives. Therefore, algorithms like Gradient Descent cannot be used. Instead, Decision Trees rely on a **Greedy Search** to evaluate every possible split.

### A. The Mathematics of Impurity
Impurity measures the chaos or mixture of classes inside a specific node (box of data). If a box is 50% Class 1 and 50% Class 0, it is at maximum impurity. If it is 100% one class, its impurity is 0.0.

**Gini Impurity (CART Default):**
Gini calculates the probability that if you blindly picked a data point from a node and randomly guessed its label based on the node's distribution, you would guess incorrectly.

$$Gini = 1 - \sum_{i=1}^{c} p_i^2$$

*(Where $c$ is the number of classes, and $p_i$ is the probability/ratio of class $i$ in that specific box).*

### B. The Update Rule: Information Gain (IG)
Information Gain is the mathematical scorecard for a Decision Tree. It measures exactly how much chaos was removed by making a specific split. 

$$IG = Gini_{parent} - \left( \frac{N_{left}}{N_{parent}} Gini_{left} + \frac{N_{right}}{N_{parent}} Gini_{right} \right)$$

### C. The Greedy Search Algorithm (How it Learns)
To find the best cut, the tree performs a brute-force search:
1. Look at the first column (Feature 1).
2. Extract all unique values in that column.
3. Treat every single unique value as a $\le$ threshold. Physically split the data into a Left Box and Right Box.
4. Calculate the Information Gain for that specific threshold.
5. Move to the next column and repeat the process for all its unique values.
6. Compare all calculated Information Gains across all columns. Lock in the Feature + Threshold combination that produced the absolute highest IG.

### D. Hand-Calculated Example 1: Sports Car Purchases

| Row | Age ($X_1$) | Salary ($X_2$) | Bought Car ($Y$) |
| :--- | :--- | :--- | :--- |
| 1 | 22 | 40000 | 0 |
| 2 | 28 | 60000 | 0 |
| 3 | 45 | 80000 | 1 |
| 4 | 20 | 20000 | 0 |
| 5 | 55 | 120000 | 1 |

*   **Parent Gini:** 3 No, 2 Yes ($N=5$). $1 - (0.6^2 + 0.4^2) = 0.48$
*   **Test Split at Age $\le$ 28:**
    *   *Left Child:* Rows 1, 2, 4. Classes = [0, 0, 0]. $Gini = 0.0$
    *   *Right Child:* Rows 3, 5. Classes = [1, 1]. $Gini = 0.0$
*   **Information Gain:** $0.48 - (0.6 \times 0.0 + 0.4 \times 0.0) = \mathbf{0.48}$ (Maximum possible IG reached; tree stops growing).

### E. Hand-Calculated Example 2: Study Hours vs Passing

| Row | Study Hours | Practice Exams | Passed Test |
| :--- | :--- | :--- | :--- |
| 1 | 2 | 0 | 0 |
| 2 | 3 | 1 | 0 |
| 3 | 8 | 3 | 1 |
| 4 | 7 | 2 | 1 |
| 5 | 9 | 4 | 1 |

*   **Parent Gini:** 2 Fail, 3 Pass ($N=5$). $1 - (0.4^2 + 0.6^2) = 0.48$
*   **Test Split at Hours $\le$ 2:**
    *   *Left Child:* 1 Row. Classes = [0]. $Gini = 0.0$
    *   *Right Child:* 4 Rows. Classes = [0, 1, 1, 1]. $Gini = 0.375$
    *   *Information Gain:* $0.48 - (0.2 \times 0.0 + 0.8 \times 0.375) = \mathbf{0.18}$
*   **Test Split at Hours $\le$ 3:**
    *   *Left Child:* 2 Rows. Classes = [0, 0]. $Gini = 0.0$
    *   *Right Child:* 3 Rows. Classes = [1, 1, 1]. $Gini = 0.0$
    *   *Information Gain:* $\mathbf{0.48}$ (This split wins over Hours $\le$ 2).

### F. Classifying New Data (Majority Voting)
Once the tree is built, mathematical impurity is no longer used. A new data point acts as a ball dropping through a flowchart. 
1. It answers the True/False rule at the Root Node.
2. It travels Left or Right based on its features.
3. It eventually lands in a terminal "Leaf Node."
4. **The Prediction:** The leaf node predicts the class that held the **Majority Vote** of the training data inside that specific box during the training phase.

### G. Computational Limitations
Decision Trees struggle with massive continuous datasets. Sorting 1 million rows and calculating the Information Gain for 1 million unique thresholds across 50 columns takes immense processing power just to draw the first root split. Modern algorithms solve this bottleneck using **Binning/Histograms** to group continuous data into discrete buckets.

---

## 5. Random Forests

A single Decision Tree is inherently unstable. If grown without limits, it will perfectly memorize the training data (severe overfitting). A tiny change in a single data point can completely alter the entire tree structure. 

Random Forests solve this using the **Wisdom of the Crowd**. Instead of one deep, unstable tree, the algorithm builds hundreds of slightly constrained trees and aggregates their predictions.

### A. Bootstrapping (Row Randomness)
To ensure the 100 trees do not just memorize the exact same patterns, they are trained on different datasets. However, the original dataset is not simply chopped into smaller pieces.
*   **Sampling with Replacement:** Every tree receives a dataset the *exact same size* as the original. Rows are drawn at random from the master dataset, copied, and "put back." 
*   **Duplicates & OOB:** Because rows are replaced, some are drawn multiple times. Mathematically, about ~63% of the data in a bootstrapped dataset is unique. The remaining ~37% of untouched rows are called **Out-Of-Bag (OOB)** data, which the Random Forest uses automatically as a free test set to validate accuracy.

### B. The Blindfold (Column Randomness)
If a dominant feature exists (e.g., Salary is the ultimate predictor), every tree in the forest will still greedily select Salary for its root split, making all the trees identical.
*   To force diversity, Random Forest enforces a **Blindfold**. 
*   Right before making a split, the algorithm randomly hides a subset of the columns. The Greedy Search is only allowed to calculate Information Gain on the remaining visible columns. 
*   This forces the forest to explore secondary and tertiary features, uncovering hidden patterns that a standard tree would ignore.

### C. Bagging
The overarching concept of a Random Forest is called **Bagging**. 
**B**ootstrapping (generating the randomized, replaced datasets) + **Agg**regating (combining all 100 leaf nodes at the end to make a final prediction via Majority Vote).

---

## 5. Evolution to Modern Boosting (The Tree Family Roadmap)

Understanding Random Forests paves the way for the most powerful tabular algorithms in Machine Learning:
*   **Gradient Boosting (XGBoost):** Moves from *Bagging* to *Boosting*. Instead of building 100 independent trees simultaneously, it builds trees sequentially. Tree 2 is mathematically engineered to correct the specific errors made by Tree 1.
*   **LightGBM:** Evolved XGBoost by introducing histogram-based binning and leaf-wise growth, allowing tree algorithms to process massive continuous datasets efficiently.
