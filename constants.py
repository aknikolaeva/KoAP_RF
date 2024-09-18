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
COLLECTION_NAME = "KoAP_RF"

# Имя модели для генерации эмбеддингов
EMBEDDING_MODEL_NAME = "ai-forever/ru-en-RoSBERTa"

# Параметры генеративной модели
TEMPERATURE = 0.25  # Температура генерации
MAX_TOKENS = 512  # Максимальное количество токенов в ответе
REPETITION_PENALTY = 1  # Штраф за повторения

SYSTEM_PROMT = "Ты русскоязычный юрист. Ответь на вопрос пользователя используя только статью. Ничего не придумывай. Не пиши ничего лишнего. Если ответа нет в статье, то напиши «не знаю»."

# Параметры сервера
# Флаг сохранения вопросов и ответов
SAVE_QA = True
