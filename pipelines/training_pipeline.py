from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.evaluation import evaluate_model
from steps.clean_data import clean_data
from steps.model_train import train_model


@pipeline
def training_pipeline(data_path):
    real_trainloader, real_testloader, height, width = ingest_data(data_path)
    clean_data(data_path)
    train_model(data_path)
    evaluate_model(data_path)
    pass