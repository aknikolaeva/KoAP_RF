import click
import chromadb

from constants import CHROMA_SETTINGS, COLLECTION_NAME, EMBEDDING_MODEL_NAME
from vector_store import VectorStore
from gpt import get_access_token, get_gpt_response
import time

from utils import log_to_csv

import logging

# Кэш для токена доступа и времени его истечения
access_token_cache = {"access_token": None, "expires_at": None}


def check_cached_access_token():
    """
    Проверяет кэшированный токен доступа и обновляет его, если он истек.
    """
    global access_token_cache
    current_time = int(time.time()) * 1000
    if (access_token_cache["expires_at"] is None) or current_time >= access_token_cache[
        "expires_at"
    ]:
        access_token_cache = get_access_token()


@click.command()
@click.option(
    "--show_sources",
    "-s",
    is_flag=True,
    help="Показывать источники вместе с ответами (По умолчанию False)",
)
@click.option(
    "--save_qa",
    is_flag=True,
    help="Сохранять пары вопросов и ответов в CSV файл (По умолчанию False)",
)
def main(show_sources, save_qa):
    """
    Основная функция для взаимодействия с пользователем, получения ответов на вопросы и логирования результатов.
    """
    vector_store = VectorStore(CHROMA_SETTINGS, EMBEDDING_MODEL_NAME)
    vector_store.init_collection(COLLECTION_NAME)

    # Интерактивные вопросы и ответы
    while True:
        query = input("\nВопрос: ")

        # Получение ответа из цепочки
        source_text = vector_store.get_most_relevant_documents(query, n_results=1)

        id = source_text["ids"][0][0]
        document = source_text["documents"][0][0]

        Query = f"Вопрос: {query}"
        Query += "\n" + f"Статья: {document}"

        check_cached_access_token()

        gpt_response = get_gpt_response(Query, access_token_cache["access_token"])
        print(f"Ответ: {gpt_response}")
        print(f"Норма: {id}")

        # Вывод релевантных источников, использованных для ответа
        if show_sources:
            print(
                "----------------------------------ИСТОЧНИК ДОКУМЕНТА---------------------------"
            )
            print(document)
            print(
                "----------------------------------ИСТОЧНИК ДОКУМЕНТА---------------------------"
            )

        # Логирование вопросов и ответов в CSV
        if save_qa:
            log_to_csv(query, gpt_response, id.split("_part")[0])

    return


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    main()
