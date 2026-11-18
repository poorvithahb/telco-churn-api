import pandas as pd
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset

def generate_drift_report():
    print("Loading data...")
    # 1. Load the original training data (Reference)
    ref_data = pd.read_csv('data/raw_data.csv')
    ref_data = ref_data.drop(columns=['customerID'])
    ref_data['TotalCharges'] = pd.to_numeric(ref_data['TotalCharges'], errors='coerce')
    ref_data['TotalCharges'] = ref_data['TotalCharges'].fillna(ref_data['TotalCharges'].median())

    # 2. Simulate "Live" data by taking a sample and introducing drift
    print("Simulating incoming live data with drift...")
    current_data = ref_data.sample(frac=0.2, random_state=42).copy()
    # Artificially double the MonthlyCharges to simulate price changes over time
    current_data['MonthlyCharges'] = current_data['MonthlyCharges'] * 2.0 

    # 3. Create the Evidently Report
    print("Generating Evidently AI Data Drift Report...")
    data_drift_report = Report(metrics=[DataDriftPreset()])
    
    # 4. Run the report comparing reference vs current
    data_drift_report.run(reference_data=ref_data, current_data=current_data)

    # 5. Save it as an HTML file so you can view it in your browser
    report_path = "monitoring/drift_report.html"
    data_drift_report.save_html(report_path)
    
    print(f"\nSuccess! Open this file to see the report: {report_path}")

if __name__ == "__main__":
    generate_drift_report()