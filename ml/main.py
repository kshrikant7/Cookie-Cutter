from ingestion.ingest import ingest_data
from input_data_quality_and_drift_checks.checks import check_data_quality, check_data_drift
from model_training.train import train_model
from model_evaluation.evaluate import evaluate_model
from prediction.predict import make_prediction
from prediction_drift.check_drift import check_prediction_drift

def main():
    current_data, reference_data = ingest_data()
    check_data_quality(reference_data, current_data)
    check_data_drift(reference_data, current_data)
    model = train_model(current_data)
    evaluate_model(model, reference_data)
    predictions = make_prediction(model, reference_data)
    check_prediction_drift(predictions)

if __name__ == "__main__":
    main()