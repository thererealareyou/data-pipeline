from form_data import DatasetMaker
from embeddings import EmbeddingService
from logging_config import setup_logging
from trainer_lora import train_lora
from trainer_full import train_full
from trainer_utils import plot_metrics_csv

if __name__ == '__main__':
    setup_logging()

    '''datasetmaker = DatasetMaker()
    datasetmaker.run_parsing_pipeline()

    embedding = EmbeddingService()
    embedding.run_matrix_pipeline()

    datasetmaker.run_triplet_pipeline()'''

    train_lora()