import re
import chromadb
from docx import Document

from constants import (
    CHROMA_SETTINGS,
    SOURCE_DOCUMENT,
    COLLECTION_NAME,
    MAX_N_TOKENS,
    STRIDE,
)

from embedder import EmbeddingGenerator

from tqdm import tqdm
from typing import List, Tuple
import logging

def get_collection(chroma_client, collection_name: str):
    """
    Проверяет существование коллекции и создает её, если она не существует.

    Параметры:
    chroma_client: Клиент для работы с коллекциями.
    collection_name (str): Имя коллекции.

    Возвращает:
    Any: Объект коллекции.
    """
    # Проверяем, существует ли уже коллекция
    collections = chroma_client.list_collections()
    if any(collection.name == collection_name for collection in collections):
        logging.info(f"Коллекция '{collection_name}' уже существует.")
        return chroma_client.get_collection(name=collection_name)

    collection = chroma_client.create_collection(name=collection_name)
    logging.info(f"Коллекция '{collection_name}' успешно создана.")
    return collection


def load_documents(doc_path: str) -> Tuple[List[str], List[str]]:
    """
    Загружает документ и разбивает его на сегменты текста, основываясь на шаблоне "Статья..."

    Параметры:
    doc_path (str): Путь к документу.

    Возвращает:
    tuple: Кортеж, содержащий список текстовых сегментов и список идентификаторов сегментов.
    """

    doc = Document(doc_path)

    text_segments = []
    text_ids = []

    segment_name = ""
    segment_txt = ""

    pattern = re.compile(r"^Статья \d+(\.\d+){0,2}(\-\d+)?\.?")

    # Итерируемся по каждому параграфу в документе
    for paragraph in doc.paragraphs:
        para = paragraph.text.strip()
        if para:
            match = pattern.match(para)
            if match:
                if segment_name:
                    text_segments.append(re.sub(pattern, "", segment_txt).strip())
                    text_ids.append(segment_name)
                segment_name = match.group(0)
                segment_txt = para
            else:
                segment_txt += "\n" + para

    return text_segments, text_ids


def main():
    """
    Основная функция для загрузки документа, разбиения его на части, генерации эмбеддингов и добавления их в коллекцию.

    Шаги:
    1. Создание клиента и получение или создание коллекции.
    2. Загрузка документа и разбиение его на части.
    3. Генерация эмбеддингов для каждой части текста.
    4. Добавление строк и эмбеддингов в коллекцию.
    """

    client = chromadb.Client(CHROMA_SETTINGS)
    collection = get_collection(client, COLLECTION_NAME)

    logging.info(f"Загрузка документа {SOURCE_DOCUMENT}")
    texts, ids = load_documents(SOURCE_DOCUMENT)
    texts_len = len(texts)
    logging.info(f"Разделено на {texts_len} частей текста")

    logging.info("Добавление строк и эмбеддингов в коллекцию")
    embed_generator = EmbeddingGenerator()

    for texts_chunk, ids_chunk in tqdm(zip(texts, ids), desc="Обработка частей"):
        documents_chunk, text_embeds_chunk = embed_generator.get_embeddings(
            texts_chunk, MAX_N_TOKENS, STRIDE
        )

        collection.add(
            documents=documents_chunk,
            ids=[ids_chunk + f"_part{pi}" for pi in range(len(text_embeds_chunk))],
            embeddings=text_embeds_chunk,
        )
    logging.info("Готово")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    main()
