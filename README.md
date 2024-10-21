# Network Intrusion Detection System

This project demonstrates the full process of training, testing, and optimizing various machine learning models, including Decision Trees, K-Nearest Neighbors (KNN), Logistic Regression, Random Forest, Naive Bayes, Support Vector Machines (SVM), using a tabular dataset.

## Table of Contents

- [Installation](#installation)
- [Data Description](#data-description)
- [Data Preprocessing](#data-preprocessing)
- [Exploratory Data Analysis (EDA)](#exploratory-data-analysis-eda)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Hyperparameter Optimization](#hyperparameter-optimization)
- [Deep Learning Models](#deep-learning-models)
- [Evaluation](#evaluation)

## Installation

To run the code, you need to install the following libraries:

```bash
pip install xgboost
pip install statsmodels
pip install pandas numpy scikit-learn matplotlib seaborn optuna tensorflow
## Data Description
The dataset consists of various network traffic records and labels them as either normal or an anomaly. The target column is **Class**, which indicates whether a data point is normal (0) or an anomaly (1).
```

- **Training data**: `traindata.csv`
- **Test data**: `testdata.csv`
- **Test labels**: `Test_labels.csv`

## Data Preprocessing
- **Missing Value Handling**: We check for missing values in both the training and test datasets and remove unnecessary columns (ID and Class).
- **Outlier Removal**: Some irrelevant or outlier services (e.g., red_i, urh_i) are removed to clean the data.
- **One-Hot Encoding**: Categorical variables like `protocol_type` and `flag` are transformed using one-hot encoding.
- **Scaling**: Feature scaling is applied to ensure that the models perform well with numerical data.

## Exploratory Data Analysis (EDA)
Visual exploration is performed to understand the relationships between features and the target label:
- Bar plots and cat plots show distributions of features like `duration`, `src_bytes`, `land`, and others.
- Correlation analysis is done to remove highly correlated features and reduce multicollinearity.

## Feature Engineering
We perform the following transformations:
- Binning numeric features (`duration`, `src_bytes`, etc.) into quantiles.
- Removing multicollinear features based on correlation analysis.
- Dimensionality reduction by eliminating features with a correlation coefficient greater than 0.9.

## Modeling
We use several machine learning algorithms:
- **Decision Tree Classifier**: A tree-based model trained with the log_loss criterion and tuned for max depth and features.
- **K-Nearest Neighbors (KNN)**: A distance-based model optimized using Optuna for hyperparameter tuning.
- **Logistic Regression**: A linear model used for binary classification.
- **Random Forest**: An ensemble model that improves accuracy through bagging.

## Hyperparameter Optimization
We use Optuna for hyperparameter tuning. For example:

```python
def objective(trial):
    dt_max_depth = trial.suggest_int('dt_max_depth', 2, 32)
    dt_max_features = trial.suggest_int('dt_max_features', 2, 10)
    classifier_obj = DecisionTreeClassifier(max_features=dt_max_features, max_depth=dt_max_depth)
    classifier_obj.fit(train_data, train_labels)
    accuracy = classifier_obj.score(test_data, test_labels)
    return accuracy
```

## Evaluation
The models are evaluated using the following metrics:

- **Accuracy**: The overall correctness of the model.
- **F1 Score**: The harmonic mean of precision and recall.
- **ROC-AUC Score**: The area under the receiver operating characteristic curve, which shows the model's ability to discriminate between the classes.

### Evaluation Metrics:

```python
F1 Score: 0.9499
Precision: 0.96
Recall: 0.90
```
