import logging

from zenml import step
class IngestData:
    '''
        Ingesting data from the data_path
    '''
    def __init__(self, data_path) -> None:
        '''
        Args:
            data_path: path to the data
        '''
        self.data_path = data_path 
    def get_data(self):
        logging.info(f"Ingesting data from {self.data_path}")
        return

@step
def ingest_data(data_path: str):
    '''
    Ingesting the data from the data path

    Args:
        data_path: path to the data
    '''
    try:
        ingest_data = IngestData(data_path)
    except:
        logging.error("Error while ingesting data: {e}")
        raise e