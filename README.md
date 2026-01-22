Heart Disease Prediction Using Machine Learning
📌 Project Overview

This project focuses on predicting the presence of heart disease using machine learning techniques.
A Logistic Regression model is trained on the Heart Disease dataset to classify whether a patient is likely to have heart disease based on medical attributes.

The system is designed using a scalable machine learning pipeline and evaluated using medically relevant performance metrics.

🛠 Tools & Technologies

Programming Language: Python 3.9

Libraries:

pandas

numpy

scikit-learn

joblib

📂 Project Structure
Heart_Disease_Project/
│
├── heart.csv                     # Dataset
├── task5.py                      # Training & evaluation script
├── heart_disease_model.pkl       # Trained model (serialized)
├── evaluation_report.txt         # Model evaluation report
├── load_model.py                 # Script to load trained model
└── README.md                     # Project documentation

📊 Dataset

Name: Heart Disease Dataset

Target Variable: target

1 → Presence of heart disease

0 → No heart disease

⚙️ Methodology

Data loading and preprocessing

Train–test split using stratified sampling

Feature scaling using StandardScaler

Model training using Logistic Regression

Cross-validation for performance stability

Model evaluation using accuracy, precision, recall, and ROC-AUC

📈 Evaluation Metrics

Accuracy

Precision

Recall (prioritized due to medical importance)

ROC–AUC

Confusion Matrix

🚀 How to Run the Project
1️⃣ Install Dependencies
py -3.9 -m pip install pandas numpy scikit-learn joblib

2️⃣ Train the Model
python task5.py


This will:

Train the model

Save the trained model as heart_disease_model.pkl

Generate evaluation_report.txt

3️⃣ Load the Trained Model
python load_model.py


Expected output:

Model loaded successfully!

📦 Deliverables

Trained Model: heart_disease_model.pkl

Evaluation Report: evaluation_report.txt

🧠 Key Highlights

Uses an end-to-end machine learning pipeline

Applies cross-validation for reliable performance

Focuses on recall to reduce false negatives

Interpretable and deployment-ready model

🔮 Future Enhancements

Try advanced models (Random Forest, XGBoost)

Feature selection and hyperparameter tuning

Deploy as a web application

✅ Conclusion

The Logistic Regression model demonstrates reliable performance for predicting heart disease.
Its high recall and ROC-AUC scores make it suitable as a baseline model for medical decision-support systems.

👤 Author

Name: Spoorthi
