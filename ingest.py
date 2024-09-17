import re
from docx import Document

from constants import (
    CHROMA_SETTINGS,
    SOURCE_DOCUMENT,
    COLLECTION_NAME,
    EMBEDDING_MODEL_NAME,
)

from vector_store import VectorStore

from tqdm import tqdm
from typing import List, Tuple
import logging

logger = logging.getLogger(__name__)


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

    """
    logging.info(f"Загрузка документа {SOURCE_DOCUMENT}")
    texts, ids = load_documents(SOURCE_DOCUMENT)
    texts_len = len(texts)
    logging.info(f"Разделено на {texts_len} частей текста")

    logging.info("Добавление строк и эмбеддингов в коллекцию")
    vector_store = VectorStore(CHROMA_SETTINGS, EMBEDDING_MODEL_NAME)
    vector_store.init_collection(COLLECTION_NAME)
    vector_store.populate_vectors(texts, ids)
    logging.info("Готово")


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )
    main()
