from zenml import pipeline
from steps.ingest_data import ingest_data

@pipeline
def training_pipeline(data_path):
    pass