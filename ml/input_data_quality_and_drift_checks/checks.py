from evidently.report import Report
from evidently.metric_preset import DataQualityPreset, DataDriftPreset

def check_data_quality(reference_data, current_data):
    print("Checking data quality...")
    data_quality_report = Report(metrics=[DataQualityPreset()])
    data_quality_report.run(reference_data=reference_data, current_data=current_data)
    data_quality_report.save_html('data_quality_report.html')

def check_data_drift(reference_data, current_data):
    print("Checking for data drift...")
    data_drift_report = Report(metrics=[DataDriftPreset()])
    data_drift_report.run(reference_data=reference_data, current_data=current_data)
    data_drift_report.save_html('data_drift_report.html')
