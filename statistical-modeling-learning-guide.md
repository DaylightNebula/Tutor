# Learning to Build Statistical Models

A reference guide for learning **supervised learning on tabular data**: models that take a fixed set of input variables (age, price, temperature, etc.) and predict one or a few outputs (a number, a category, a probability).

---

## Part 1: Learning Path

### 1. Get comfortable with the basics (only as much as you need)
- **Probability:** distributions, mean/variance, conditional probability
- **Linear algebra:** vectors, matrices, matrix multiplication
- **Calculus:** the idea of a derivative / gradient

Don't spend months here. Learn alongside the models and revisit when something doesn't make sense.

### 2. Start with linear regression, and build it yourself
The single most important model to understand deeply. Implement it from scratch in Python with NumPy **twice**:
1. Closed-form solution (normal equation)
2. Gradient descent

Once "minimize the error by adjusting the weights" clicks, almost every other model is a variation on it.

### 3. Learn the core ideas that apply to every model
- **Train/test splits and cross-validation:** always judge a model on data it didn't see while training
- **Overfitting vs. underfitting:** the bias–variance tradeoff
- **Loss functions:** what you're actually optimizing
- **Regularization (ridge, lasso):** penalizing complexity so the model generalizes
- **Evaluation metrics:**
  - Regression: RMSE, MAE, R²
  - Classification: accuracy, precision/recall, AUC

### 4. Work through the standard toolkit (roughly in order)
1. **Logistic regression:** classification; a bridge to neural nets
2. **k-nearest neighbors:** simple, intuitive, non-parametric
3. **Decision trees → random forests → gradient boosting** (XGBoost / LightGBM). Gradient boosting is often the best performer on tabular data.
4. **Small neural networks (multilayer perceptrons):** still finite-in, finite-out. Build a tiny one from scratch to see what's under the hood.
5. **Bayesian modeling (optional):** models that express uncertainty and encode your assumptions

### 5. Use real libraries once you understand the mechanics
- `scikit-learn`: most ML models
- `statsmodels`: classical statistics (p-values, confidence intervals)
- `PyMC`: Bayesian modeling

### Recommended resources
- **An Introduction to Statistical Learning (ISL / ISLP)** by James, Witten, Hastie & Tibshirani. Free online, with a Python edition. The best single book for this.
- **Statistical Rethinking** by Richard McElreath. Builds models from first principles (Bayesian). Free lecture videos.
- **Andrew Ng's Machine Learning Specialization** (Coursera). Video lectures with guided assignments.
- **Kaggle.** Datasets, beginner competitions (Titanic, House Prices), and community notebooks.

### The thing that matters most
Pick a small problem you actually care about and model it. Real, messy data teaches what textbooks skip: cleaning, feature choice, and diagnosing why a model is confidently wrong.

**A good rhythm:** read a chapter → build the model from scratch on a toy dataset → use the library version on your real problem → compare results.

---

## Part 2: Datasets on Hugging Face

Hugging Face's own `scikit-learn` organization (https://huggingface.co/scikit-learn) hosts small, clean tabular CSVs that fit this path well.

### Stage 1: First models (tiny, clean, build-from-scratch)
| Dataset | Rows | Task | Notes |
|---|---|---|---|
| `scikit-learn/iris` | 150 | Classification | 4 features, 3 flower species. Small enough to verify by hand. |
| `scikit-learn/tips` | 244 | Regression | Predict tip from bill, party size, etc. Great first linear regression. |
| `scikit-learn/Fish` | 159 | Regression | Predict weight from measurements. Try polynomial/log transforms. |
| `scikit-learn/auto-mpg` | 398 | Regression | Predict fuel efficiency from car specs. |

### Stage 2: Classification and evaluation metrics
| Dataset | Rows | Task | Notes |
|---|---|---|---|
| `scikit-learn/breast-cancer-wisconsin` | 569 | Binary classification | 30 features. Learn why precision/recall matter more than accuracy. |
| `julien-c/titanic-survival` or `mstz/titanic` | ~891 | Binary classification | Messy columns (Name, Ticket, Cabin, Embarked) and missing ages. Practice cleaning and feature engineering. |

### Stage 3: Mid-size, realistic (trees, ensembles, regularization)
| Dataset | Rows | Task | Notes |
|---|---|---|---|
| `scikit-learn/adult-census-income` | 32,561 | Binary classification | Predict income > $50K. Mixed numeric + categorical. Compare logistic regression vs. random forest vs. gradient boosting. |
| `gvlassis/california_housing` | 20,640 | Regression | 1990 California census; target is median house value. Benchmark linear vs. trees vs. small neural net. |
| `scikit-learn/churn-prediction` | ~7,000 | Binary classification | Business problem with imbalanced classes. |
| `scikit-learn/credit-card-clients` | 30,000 | Binary classification | Predicting default; imbalanced classes. |

### Loading a dataset
```python
from datasets import load_dataset

df = load_dataset("scikit-learn/iris")["train"].to_pandas()
```
The result is a regular pandas DataFrame, ready for NumPy, scikit-learn, or statsmodels.

### Suggested order
1. **tips** or **Fish**: linear regression from scratch
2. **iris**: classification from scratch
3. **breast-cancer-wisconsin**: evaluation metrics
4. **Titanic**: data cleaning and feature engineering
5. **adult-census-income** or **california_housing**: compare model families

### Tips
- Many Hugging Face datasets are community re-uploads of the same classics. If one copy has odd formatting, try another.
- The same datasets are also on **Kaggle** and the **UCI Machine Learning Repository** (https://archive.ics.uci.edu).
