import chromadb
from sentence_transformers import SentenceTransformer
import logging
from typing import List, Tuple

logger = logging.getLogger(__name__)


class VectorStore:
    """
    Класс для работы с векторным хранилищем на основе ChromaDB и модели SentenceTransformer.
    """

    def __init__(self, chroma_settings: dict, embedding_model_name: str):
        """
        Инициализация класса VectorStore.

        Параметры:
        chroma_settings (dict): Настройки для клиента ChromaDB.
        embedding_model_name (str): Имя модели для создания эмбеддингов.
        """
        self.chroma_client = chromadb.Client(chroma_settings)
        self.embedding_model = SentenceTransformer(embedding_model_name)
        self.collection = None

    def init_collection(self, collection_name: str):
        """
        Проверяет существование коллекции и создает её, если она не существует.

        Параметры:
        collection_name (str): имя коллекции.
        """
        # Проверяем, существует ли уже коллекция
        collections = self.chroma_client.list_collections()
        if any(collection.name == collection_name for collection in collections):
            self.collection = self.chroma_client.get_collection(name=collection_name)
            logging.info(f"Коллекция уже существует.")
        else:
            self.collection = self.chroma_client.create_collection(name=collection_name)
            logging.info(f"Коллекция успешно создана.")

    def populate_vectors(self, texts: List[str], ids: List[str]):
        """
        Метод для заполнения векторного хранилища эмбеддингами из набора данных.

        Параметры:
        texts (List[str]): Список текстов для создания эмбеддингов.
        ids (List[str]): Список идентификаторов для текстов.
        """
        for text, idx in zip(texts, ids):
            embeddings = self.embedding_model.encode(text).tolist()
            self.collection.add(embeddings=[embeddings], documents=[text], ids=[idx])

    def get_most_relevant_documents(self, query_text: str, n_results: int = 3):
        """
        Метод поиска в коллекции ChromaDB соответствующего контекста на основе запроса.

        Параметры:
        query_text (str): Текст запроса.
        n_results (int): Количество результатов для возврата.
        """
        # Подготовка эмбеддинга запроса
        query_embeddings = self.embedding_model.encode(query_text).tolist()

        # Запрос к коллекции
        results = self.collection.query(
            query_embeddings=query_embeddings, n_results=n_results
        )
        return results
