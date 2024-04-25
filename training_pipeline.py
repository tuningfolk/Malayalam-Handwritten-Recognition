# from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.evaluation import evaluate_model
from steps.clean_data import clean_data
from steps.model_train import train_model

# from steps import clean_data

# @pipeline
def training_pipeline(data_path):
    data = ingest_data(data_path)
    print(dir(data))
    # print(data.annotation)
    real_trainloader, real_testloader, height, width = data
    clean_data(data)
    train_model(data_path)
    evaluate_model(data_path)
    pass
