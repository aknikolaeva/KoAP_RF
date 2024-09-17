import os
from chromadb.config import Settings

ROOT_DIRECTORY = os.path.dirname(os.path.realpath(__file__))

# Путь к исходному документу
SOURCE_DOCUMENT = f"{ROOT_DIRECTORY}/SOURCE_DOCUMENTS/KoAP_RF.docx"

# Директория для сохранения базы данных
PERSIST_DIRECTORY = f"{ROOT_DIRECTORY}/DB"

# Настройки для ChromaDB
CHROMA_SETTINGS = Settings(
    anonymized_telemetry=False, is_persistent=True, persist_directory=PERSIST_DIRECTORY
)

# Имя коллекции в базе данных
COLLECTION_NAME = "KoAP_RF_test_v2.99"

# Имя модели для генерации эмбеддингов
EMBEDDING_MODEL_NAME = "ai-forever/ru-en-RoSBERTa"
