## 🚀 Key Features

### 📊 Exploratory Data Analysis (EDA)
* **Univariate & Bivariate Analysis:** Visualizing distributions of tenure, monthly charges, and payment methods.
* **Correlation Heatmaps:** Identifying key drivers of churn.
* **Churn Profiling:** Understanding the demographics of customers who leave.

### 🧠 Machine Learning Pipeline
* **Data Preprocessing:** Handling missing values, encoding categorical variables (One-Hot/Label Encoding), and Feature Scaling.
* **Imbalance Handling:** Utilized techniques like **SMOTE** (Synthetic Minority Over-sampling Technique) to handle class imbalance.
* **Model Selection:** Compared multiple algorithms including:
    * Logistic Regression
    * Random Forest Classifier
    * XGBoost / Gradient Boosting
* **Hyperparameter Tuning:** Optimized model performance using GridSearchCV.

### 📈 Metrics & Evaluation
* Focused on **Recall** and **F1-Score** to minimize False Negatives (missing a customer who is actually about to churn).
* **ROC-AUC Score:** Evaluated the model's ability to distinguish between classes.

---

## 🛠️ Tech Stack

| Domain | Tools/Libraries |
| :--- | :--- |
| **Language** | Python |
| **Data Manipulation** | Pandas, NumPy |
| **Visualization** | Matplotlib, Seaborn |
| **Machine Learning** | Scikit-Learn, XGBoost |
| **Notebook Env** | Jupyter Notebook / Google Colab |
| **Deployment (Optional)** | Streamlit / Flask |

---

## 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| Logistic Regression | 79% | 0.65 | 0.72 | 0.68 |
| **Random Forest** | **85%** | **0.78** | **0.81** | **0.79** |
| XGBoost | 84% | 0.76 | 0.79 | 0.77 |

*> **Note:** The Random Forest model was selected as the final model due to its balance between precision and recall.*

---
