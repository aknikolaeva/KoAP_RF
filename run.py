import click
import chromadb

from constants import CHROMA_SETTINGS, COLLECTION_NAME, MAX_N_TOKENS, STRIDE

from embedder import EmbeddingGenerator
from gpt import get_gpt_response


import json
import requests
import time

from utils import log_to_csv, load_config

import logging

secrets = load_config("config/secrets.yaml")

embed_generator = EmbeddingGenerator()


def get_most_relevant_documents(query_text: str, n_results: int = 3):
    """
    Получает наиболее релевантные документы для заданного запроса.
    """

    client = chromadb.Client(CHROMA_SETTINGS)

    # Получение коллекции
    collection = client.get_collection(COLLECTION_NAME)

    # Подготовка эмбеддинга запроса
    _, query_embedding = embed_generator.get_embeddings(query_text, MAX_N_TOKENS, STRIDE, 
                                                        min_chunk_len=0)

    # Запрос к коллекции
    results = collection.query(query_embeddings=query_embedding, n_results=n_results)

    # Возвращение результатов
    return results


# Кэш для токена доступа и времени его истечения
access_token_cache = {"access_token": None, "expires_at": None}


def get_access_token() -> dict:
    """
    Получает токен доступа от API.

    Returns:
        dict: Словарь с токеном доступа и временем его истечения.
    """

    url = "https://ngw.devices.sberbank.ru:9443/api/v2/oauth"
    payload = "scope=GIGACHAT_API_PERS"
    headers = {
        "Content-Type": "application/x-www-form-urlencoded",
        "Accept": "application/json",
        "RqUID": secrets["client_id"],
        "Authorization": f"Basic {secrets['authorization_data']}",
    }
    response = requests.request(
        "POST", url, headers=headers, data=payload, verify="./russiantrustedca.pem"
    )
    return json.loads(response.text)


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

    # Интерактивные вопросы и ответы
    while True:
        query = input("\nВопрос: ")
        if query == "exit":
            break
        # Получение ответа из цепочки
        source_text = get_most_relevant_documents(query, n_results=1)

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
