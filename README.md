Autism Prediction Model 🧠
This project is focused on predicting the likelihood of autism in individuals based on a variety of features, using machine learning algorithms. The model is trained using a cleaned dataset and leverages classification techniques like Decision Tree, Random Forest, and XGBoost for prediction accuracy.
 
📁 Project Structure
autismprediction.ipynb – Main Jupyter Notebook with full data analysis, preprocessing, model training, evaluation, and export.

trainautismprediction.csv – Dataset used for model training (referenced in the notebook, not included in this repo).

model.pkl – Pickled trained model (can be used for deployment).

📊 Features Used
Gender

Ethnicity

Jaundice at birth

Family history

Age

Screening test scores

And more categorical features...

✅ Key Steps
Data Preprocessing

Removal of unnecessary columns (ID, age_desc)

Label encoding of categorical features

Handling class imbalance with SMOTE

Modeling

Models used: DecisionTreeClassifier, RandomForestClassifier, XGBClassifier

Hyperparameter tuning with RandomizedSearchCV

Evaluation via accuracy, confusion matrix, classification report

Model Export

Final model saved using pickle for deployment purposes

📈 Model Performance
Achieved high accuracy with XGBoost

Class imbalance handled using SMOTE for fair evaluation

Visualizations included to support EDA and feature importance

🛠 Libraries Used
pandas, numpy, matplotlib, seaborn

scikit-learn, xgboost, imblearn

pickle

Here's a clean and professional version of your "How to Run" section for the `README.md` — ready to copy and paste into your GitHub README:

---

## 🚀 How to Run

### 1. Clone the Repository
git clone https://github.com/yourusername/autism-prediction.git
cd autism-prediction
### 2. Install Dependencies
Make sure you have Python and pip installed. 
### 3. Launch the Jupyter Notebook
 jupyter notebook autismprediction.ipynb


