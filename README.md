# Student Performance Analysis - Linear Regression

A machine learning project that predicts student final grades (G3) using Linear Regression based on various demographic, social, and academic features.

## 📊 Project Overview

This project analyzes student performance data from two Portuguese schools (Gabriel Pereira and Mousinho da Silveira) to predict final grades. The analysis explores how factors like parental education, study time, failures, alcohol consumption, and previous grades influence student academic outcomes.

**Key Achievement**: Built a predictive model that accurately forecasts student final grades with strong performance metrics.

## 📁 Dataset

The `student_data.csv` dataset contains information about students including:

- **Demographic**: age, sex, address (urban/rural), family size
- **Social**: parent's education and occupation, family relationships, going out frequency
- **Academic**: study time, failures, school support, grades (G1, G2, G3)
- **Behavioral**: alcohol consumption (weekday/weekend), absences, romantic relationships

**Target Variable**: G3 (final grade, ranging from 0 to 20)

## 🛠️ Installation & Requirements

### Prerequisites

```bash
Python 3.7+
```

### Required Libraries

```bash
pip install pandas numpy matplotlib seaborn scikit-learn ydata-profiling
```

**Core Dependencies**:

- `pandas` - Data manipulation and analysis
- `numpy` - Numerical computations
- `matplotlib` & `seaborn` - Data visualization
- `scikit-learn` - Machine learning algorithms and preprocessing
- `ydata-profiling` - Automated EDA report generation

## 📂 Project Structure

```
Linear Regression-Student Performance Analysis/
│
├── student_data.csv                                      # Dataset
├── Performance_Prediction_Linear_Regression.ipynb        # Main analysis notebook
├── useModel.ipynb                                        # Model usage/testing notebook
├── output.html                                           # EDA profiling report
├── PerformancePrediction.pkl                             # Trained model (generated)
└── README.md                                             # Project documentation
```

## 🔍 Analysis Workflow

### 1. **Exploratory Data Analysis (EDA)**

- Generated comprehensive profiling report using `ydata-profiling`
- Analyzed grade distributions across different schools (GP vs MS)
- Examined performance patterns by gender, address type, and parental status
- **Key Finding**: MS school students showed lower average grades compared to GP school

### 2. **Data Preprocessing**

- **One-Hot Encoding**: Converted categorical variables (school, sex, address, parent jobs, etc.) into numerical format
- **Feature Engineering**: Applied log transformation (`log1p`) to the absences column to normalize its distribution
- **Feature Selection**: Evaluated all features and their correlations with target variable

### 3. **Model Development**

The project explored multiple modeling approaches:

#### Initial Approach (Excluding G1 & G2)

- **Features**: 18 selected features with moderate correlation
- **Result**: Poor performance (R² ~ 0.08, MAE ~ 3.34)
- **Insight**: G1 and G2 are crucial predictors, not data leakage

#### Final Model (Including All Features)

- **Features**: All 40+ features after encoding, including G1 and G2
- **Preprocessing Pipeline**:
  1. One-Hot Encoding for categorical variables
  2. Log transformation on absences column
  3. Linear Regression model

### 4. **Model Evaluation**

The final model was evaluated using:

- **Mean Absolute Error (MAE)**: Average prediction error
- **Mean Squared Error (MSE)**: Squared prediction errors
- **R² Score**: Proportion of variance explained
- **10-Fold Cross-Validation**: Ensures model generalization

## 📈 Results

### Final Model Performance

The Linear Regression model with complete feature set achieved strong predictive performance:

- **Excellent correlation** with actual grades
- **Robust cross-validation scores** indicating good generalization
- **Key Predictors**: G1 (first period grade), G2 (second period grade), failures, study time, parental education

### Feature Importance Insights

Based on correlation analysis:

**Strong Positive Correlations**:

- Previous grades (G1, G2) - strongest predictors
- Mother's education (Medu)
- Father's education (Fedu)
- Aspiration for higher education

**Negative Correlations**:

- Number of past failures
- Going out frequency
- Weekday alcohol consumption (Dalc)
- Absences

### Visualizations

The notebook includes:

- Distribution plots for grades by school type
- Gender-based performance comparisons
- Address type (urban vs rural) analysis
- Parental status impact visualization
- Correlation heatmaps for feature selection
- KDE plots for numerical feature distributions

## 🚀 Usage

### Loading the Trained Model

```python
import pickle
import pandas as pd

# Load the model
model = pickle.load(open('PerformancePrediction.pkl', 'rb'))

# Prepare input data (same format as training data)
student_data = pd.DataFrame({
    'school': ['GP'],
    'sex': ['F'],
    'age': [17],
    # ... include all other features
    'G1': [15],
    'G2': [14]
})

# Make prediction
predicted_grade = model.predict(student_data)
print(f"Predicted Final Grade: {predicted_grade[0]:.2f}")
```

See [useModel.ipynb](useModel.ipynb) for detailed usage examples.

## 🎯 Key Takeaways

1. **Previous academic performance** (G1, G2) is the strongest predictor of final grades
2. **Parental education** and student **aspiration for higher education** positively impact performance
3. **Failures, absences, and alcohol consumption** negatively correlate with academic success
4. **Study time** shows positive correlation with better grades
5. School type matters - GP school students generally perform better than MS school students

## 🔮 Future Improvements

- [ ] Experiment with ensemble methods (Random Forest, Gradient Boosting)
- [ ] Implement feature importance analysis
- [ ] Add polynomial features for non-linear relationships
- [ ] Create interactive visualization dashboard
- [ ] Develop early warning system for at-risk students
- [ ] Compare multiple regression algorithms
- [ ] Add hyperparameter tuning using GridSearchCV

## 📝 License

This project is available for educational purposes.

## 👤 Author

**Nirvan**

📚 Data Source
Dataset: Student Performance Dataset
Description: Student Performance Dataset with Detailed and Variety of (33) Features
Author: Dev Ansodariya
Platform: Kaggle

_This project demonstrates end-to-end machine learning workflow including EDA, preprocessing, feature engineering, model training, and evaluation._
