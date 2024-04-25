# from zenml import pipeline
from steps.ingest_data import ingest_data
from steps.evaluation import evaluate_model
from steps.clean_data import clean_data
from steps.model_train import train_model
import torch
# from steps import clean_data

# @pipeline
def training_pipeline(data_path):
    data = ingest_data(data_path)
    real_trainloader, real_testloader,\
          char_to_label,label_to_char, num_classes = data
    clean_data(data)
    model = train_model(data, num_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("DEVICE: ", device)
    loss = evaluate_model(real_trainloader,real_testloader,model,device)
    print(loss)
    pass
