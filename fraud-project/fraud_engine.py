import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
print("==================================================")
print("   FINANCIAL FRAUD INTELLIGENCE ENGINE STARTING   ")
print("==================================================")
print("\n[1] Generating realistic transaction data...")
np.random.seed(42)
normal = pd.DataFrame(np.random.normal(loc=50, scale=10, size=(5000, 5)), columns=['V1','V2','V3','V4','Amount'])
normal['Class'] = 0
fraud = pd.DataFrame(np.random.normal(loc=80, scale=25, size=(100, 5)), columns=['V1','V2','V3','V4','Amount'])
fraud['Class'] = 1
df = pd.concat([normal, fraud]).sample(frac=1).reset_index(drop=True)
print("[2] Training Unsupervised Anomaly Detection (Isolation Forest)...")
iso_forest = IsolationForest(contamination=0.02, random_state=42)
df['Anomaly_Score'] = iso_forest.fit_predict(df[['V1','V2','V3','V4','Amount']])
df['Anomaly'] = df['Anomaly_Score'].apply(lambda x: 1 if x == -1 else 0)
print("[3] Training Supervised Risk Scoring Model (Random Forest)...")
X = df.drop(columns=['Class', 'Anomaly_Score', 'Anomaly'])
y = df['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
rf_model = RandomForestClassifier(random_state=42)
rf_model.fit(X_train_scaled, y_train)
risk_scores = rf_model.predict_proba(X_test_scaled)[:, 1]
print("\n--- MODEL EVALUATION RESULTS ---")
print(classification_report(y_test, rf_model.predict(X_test_scaled)))
print("[4] Generating Risk Scoring Dashboard...")
plt.figure(figsize=(14, 6))
plt.subplot(1, 2, 1)
sns.histplot(risk_scores, bins=30, kde=True, color='crimson')
plt.title('Risk Score Distribution (Fraud Probability)')
plt.xlabel('Risk Score (0.0 = Safe, 1.0 = High Risk)')
plt.ylabel('Number of Transactions')
plt.subplot(1, 2, 2)
cm = confusion_matrix(y_test, rf_model.predict(X_test_scaled))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix (Actual vs Predicted)')
plt.xlabel('Predicted Label')
plt.ylabel('Actual Label')
plt.tight_layout()
plt.savefig('risk_dashboard.png')
print("\n==================================================")
print("SUCCESS! Dashboard saved as 'risk_dashboard.png'.")
print("==================================================")