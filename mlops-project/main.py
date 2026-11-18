from src.train import train_models
from monitoring.monitor import generate_drift_report

print("========================================")
print("   STARTING END-TO-END MLOPS PIPELINE   ")
print("========================================")

print("\n[1/2] Running Training Pipeline...")
train_models('data/raw_data.csv')

print("\n[2/2] Running Monitoring Pipeline...")
generate_drift_report()

print("\n========================================")
print("   PIPELINE COMPLETE. READY FOR DEPLOY. ")
print("========================================")