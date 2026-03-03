# 📊 Gaussian Discriminant Analysis: LDA & QDA from Scratch

This project implements **Bayesian Decision Theory** using two-dimensional Gaussian distributions. Rather than relying on high-level machine learning libraries for the model logic, this repository features a from-scratch implementation of **Linear Discriminant Analysis (LDA)** and **Quadratic Discriminant Analysis (QDA)**.

## 🔬 Project Overview
This project is a deep dive into Statistical Pattern Recognition. It demonstrates the implementation of a Two-Class Bayesian Classifier designed to distinguish between synthetic data distributions. By generating 2D Gaussian datasets and deriving Linear (LDA) and Quadratic (QDA) discriminant functions from scratch, this project explores how a machine uses probability density to define optimal decision boundaries.


### 🛠️ Technical Implementation
* **Synthetic Data Generation:** Uses **Cholesky Decomposition** to transform standard normal samples into multivariate Gaussian distributions with specific mean vectors and covariance matrices.
* **Discriminant Functions:**
    * **Quadratic (QDA):** Implements the full log-likelihood function, accounting for class-specific covariance determinants ($| \Sigma_i |$).
    * **Linear (LDA):** Implements the simplified linear classifier by calculating a **pooled covariance matrix** across classes.
* **Verification:** Evaluates model performance using **Confusion Matrices**, **F1-Scores**, and **Balanced Accuracy** via `scikit-learn` metrics.

## 📊 Visualizing Boundaries
The project generates decision surfaces to show the difference between the linear boundary (where covariances are assumed equal) and the quadratic boundary (where they are allowed to differ).


## 🧪 How to Run
1. Ensure you have `numpy`, `matplotlib`, `seaborn`, and `scikit-learn` installed.
2. Open the Jupyter Notebook: `Bayesian_Classification.ipynb`.
3. Run all cells to see the data generation, training, and boundary visualization.
