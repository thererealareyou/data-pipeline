from form_data import DatasetMaker
from embeddings import EmbeddingService
from logging_config import setup_logging

if __name__ == '__main__':
    setup_logging()

    datasetmaker = DatasetMaker()
    datasetmaker.run_parsing_pipeline()

    embedding = EmbeddingService()
    embedding.run_matrix_pipeline()

    datasetmaker.run_triplet_pipeline()