from pydantic import BaseModel

from constants import CHROMA_SETTINGS, COLLECTION_NAME, EMBEDDING_MODEL_NAME, SAVE_QA
from core.vector_store import VectorStore
from core.gpt import get_gpt_response
from core.utils import log_to_csv
from core.handler import AccessTokenManager

access_token_manager = AccessTokenManager()


class QueryRequest(BaseModel):
    query: str
    show_sources: bool = False


class Retriever:

    def __init__(
        self,
    ):

        # Инициализация векторного хранилища
        self.vector_store = VectorStore(CHROMA_SETTINGS, EMBEDDING_MODEL_NAME)
        self.vector_store.init_collection(COLLECTION_NAME)

        # Менеджер доступа
        self.access_token_manager = access_token_manager

    async def ask(self, request: QueryRequest):
        """
        Метод для обработки запроса и получения ответа.
        """

        query = request.query

        # Получение наиболее релевантного документа из векторного хранилища
        source_text = self.vector_store.get_most_relevant_documents(query, n_results=1)

        # Получение идентификатора и текста документа
        id = source_text["ids"][0][0]
        document = source_text["documents"][0][0]

        # Формирование запроса для модели GPT
        Query = f"Вопрос: {query}"
        Query += "\n" + f"Статья: {document}"

        # Получение токена доступа
        access_token = self.access_token_manager.get_access_token()

        # Получение ответа от модели GPT
        gpt_response = get_gpt_response(Query, access_token)

        response = {
            "answer": gpt_response,
            "norm": id,
        }

        # Добавление исходного документа в ответ, если требуется
        if request.show_sources:
            response["source_document"] = document

        # Логирование вопросов и ответов в CSV, если включено
        if SAVE_QA:
            log_to_csv(query, gpt_response, id.split("_part")[0])

        return response
