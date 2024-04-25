import logging


def clean_data(data):
    print(type(data))

    real_trainloader, real_testloader, char_to_label, label_to_char, num_classes = data
   