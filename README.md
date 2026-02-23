# 💼 Employee Salary Prediction - HR Analytics

A **machine learning project predicting employee salaries** based on experience, job role, education, and performance metrics for HR analytics and compensation planning.

## 🎯 Overview

This project provides:
- ✅ Salary prediction from employee data
- ✅ Experience-based feature engineering
- ✅ Job role & department analysis
- ✅ Regression models for salary forecasting
- ✅ Compensation fairness analysis
- ✅ Salary band recommendations

## 📊 Employee Dataset

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class EmployeeSalaryAnalysis:
    """Analyze employee salary data"""
    
    def __init__(self, filepath='employee_salary.csv'):
        self.df = pd.read_csv(filepath)
    
    def explore_salaries(self):
        """Dataset overview"""
        print(f"Total employees: {len(self.df)}")
        print(f"\nSalary statistics:")
        print(self.df['Salary'].describe())
        
        # By department
        print(f"\nAverage salary by department:")
        print(self.df.groupby('Department')['Salary'].mean().sort_values(ascending=False))
        
        # By role
        print(f"\nAverage salary by role:")
        print(self.df.groupby('JobRole')['Salary'].mean().sort_values(ascending=False))
    
    def salary_distribution(self):
        """Analyze distribution"""
        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        
        # Overall distribution
        axes[0, 0].hist(self.df['Salary'], bins=30, edgecolor='black')
        axes[0, 0].set_title('Salary Distribution')
        
        # By department
        for dept in self.df['Department'].unique():
            dept_data = self.df[self.df['Department'] == dept]['Salary']
            axes[0, 1].hist(dept_data, alpha=0.5, label=dept, bins=20)
        axes[0, 1].set_title('Salary by Department')
        axes[0, 1].legend()
        
        # By experience
        axes[1, 0].scatter(self.df['YearsOfExperience'], self.df['Salary'], alpha=0.6)
        axes[1, 0].set_xlabel('Years of Experience')
        axes[1, 0].set_ylabel('Salary')
        axes[1, 0].set_title('Experience vs Salary')
        
        # By education
        self.df.boxplot(column='Salary', by='EducationLevel', ax=axes[1, 1])
        axes[1, 1].set_title('Salary by Education')
        
        plt.tight_layout()
        plt.show()
```

## 🔧 Feature Engineering

```python
class EmployeeFeatureEngineer:
    """Create HR features"""
    
    @staticmethod
    def experience_features(df):
        """Experience-based features"""
        df_copy = df.copy()
        
        # Years of experience categories
        df_copy['Experience_Level'] = pd.cut(df_copy['YearsOfExperience'],
                                             bins=[0, 2, 5, 10, 20, 50],
                                             labels=['Entry', 'Junior', 'Mid', 'Senior', 'Executive'])
        
        # Experience scaled (log for diminishing returns)
        df_copy['Experience_Log'] = np.log1p(df_copy['YearsOfExperience'])
        
        # Experience squared (non-linear relationship)
        df_copy['Experience_Squared'] = df_copy['YearsOfExperience'] ** 2
        
        return df_copy
    
    @staticmethod
    def performance_features(df):
        """Performance-based features"""
        df_copy = df.copy()
        
        # Performance bonus potential
        df_copy['Bonus_Score'] = df_copy['PerformanceRating'] * df_copy['YearsOfExperience']
        
        # Promotion readiness
        df_copy['PromotionReady'] = ((df_copy['PerformanceRating'] >= 4) & 
                                     (df_copy['YearsOfExperience'] >= 3)).astype(int)
        
        # Skill value
        df_copy['SkillValue'] = df_copy['SkillsCertifications'] * 0.75 + \
                               df_copy['PerformanceRating'] * 0.25
        
        return df_copy
    
    @staticmethod
    def department_features(df):
        """Department & role features"""
        df_copy = df.copy()
        
        # Department median salary (for comparison)
        dept_median = df_copy.groupby('Department')['Salary'].median()
        df_copy['Dept_Median_Salary'] = df_copy['Department'].map(dept_median)
        
        # Role market rate
        role_median = df_copy.groupby('JobRole')['Salary'].median()
        df_copy['Role_Market_Rate'] = df_copy['JobRole'].map(role_median)
        
        # Seniority in department
        df_copy['Seniority_Rank'] = df_copy.groupby('Department')['YearsOfExperience'].rank()
        
        return df_copy
    
    @staticmethod
    def categorical_encoding(df):
        """Encode categorical features"""
        from sklearn.preprocessing import LabelEncoder
        
        df_copy = df.copy()
        
        le_dept = LabelEncoder()
        le_role = LabelEncoder()
        le_edu = LabelEncoder()
        
        df_copy['Department_Encoded'] = le_dept.fit_transform(df_copy['Department'])
        df_copy['JobRole_Encoded'] = le_role.fit_transform(df_copy['JobRole'])
        df_copy['Education_Encoded'] = le_edu.fit_transform(df_copy['EducationLevel'])
        
        return df_copy
```

## 🤖 Regression Models

```python
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

class SalaryPredictor:
    """Predict employee salaries"""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.models = self._build_models()
    
    def _build_models(self):
        """Initialize models"""
        return {
            'Linear Regression': LinearRegression(),
            'Ridge Regression': Ridge(alpha=100.0),
            'Lasso Regression': Lasso(alpha=1000.0),
            'Random Forest': RandomForestRegressor(n_estimators=100, max_depth=15, random_state=42),
            'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, learning_rate=0.05, random_state=42)
        }
    
    def train_ensemble(self, X_train, y_train):
        """Train all models"""
        trained = {}
        X_scaled = self.scaler.fit_transform(X_train)
        
        for name, model in self.models.items():
            if name in ['Ridge Regression', 'Lasso Regression']:
                model.fit(X_scaled, y_train)
            else:
                model.fit(X_train, y_train)
            
            trained[name] = model
        
        self.models = trained
        return trained
    
    def predict_salary(self, employee_features):
        """Predict salary for employee"""
        predictions = {}
        
        for model_name, model in self.models.items():
            pred = model.predict(employee_features.reshape(1, -1))
            predictions[model_name] = pred[0]
        
        # Ensemble average
        avg_prediction = np.mean(list(predictions.values()))
        
        return {
            'Individual_Predictions': predictions,
            'Ensemble_Prediction': avg_prediction,
            'Range': (avg_prediction * 0.9, avg_prediction * 1.1)
        }
```

## 📊 Salary Analytics

```python
class SalaryAnalytics:
    """Analyze compensation"""
    
    @staticmethod
    def salary_equity_analysis(df):
        """Check for salary equity issues"""
        if 'Gender' in df.columns:
            print("Salary gap by gender:")
            gender_salary = df.groupby('Gender')['Salary'].agg(['mean', 'median', 'count'])
            print(gender_salary)
            
            gap = (1 - gender_salary.loc[df['Gender'].unique()[1], 'mean'] / 
                   gender_salary.loc[df['Gender'].unique()[0], 'mean']) * 100
            print(f"\nSalary gap: {gap:.2f}%")
    
    @staticmethod
    def salary_bands(df):
        """Generate salary bands"""
        salary_ranges = df.groupby('JobRole')['Salary'].agg(['min', 'mean', 'max', 'std']).round(0)
        
        salary_ranges['Lower_Band'] = (salary_ranges['mean'] - salary_ranges['std']).astype(int)
        salary_ranges['Upper_Band'] = (salary_ranges['mean'] + salary_ranges['std']).astype(int)
        
        print("\nRecommended Salary Bands by Role:")
        print(salary_ranges[['Lower_Band', 'mean', 'Upper_Band']])
        
        return salary_ranges
    
    @staticmethod
    def performance_salary_correlation(df):
        """Analyze performance-pay relationship"""
        print("\nPerformance vs Salary Correlation:")
        correlation = df[['PerformanceRating', 'YearsOfExperience', 'Salary']].corr()
        print(correlation['Salary'].sort_values(ascending=False))
```

## 💡 Interview Talking Points

**Q: Address salary gaps?**
```
Answer:
- Identify undercompensated employees
- Industry benchmarking
- Role-based salary bands
- Performance-based adjustments
- Gradual correction plan
```

**Q: Factors impacting salary?**
```
Answer:
- Years of experience (strongest)
- Performance rating
- Education level
- Job role & specialty
- Department/location
```

## 🌟 Portfolio Value

✅ HR analytics
✅ Salary prediction
✅ Compensation analysis
✅ Equity assessment
✅ Feature engineering
✅ Regression modeling
✅ Business decision support

---

**Technologies**: Scikit-learn, Pandas, NumPy, Matplotlib

