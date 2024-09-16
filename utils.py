import os
import yaml
import csv
from datetime import datetime
from typing import Any, List


def load_config(file_path: str):
    """
    Загружает конфигурацию из YAML файла.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def flatten(nested_list: List[List[Any]]):
    """
    Преобразует вложенный список в плоский список.
    """
    flat_list = []
    for sublist in nested_list:
        for item in sublist:
            flat_list.append(item)
    return flat_list


def log_to_csv(question: str, answer: str, norma: str):
    """
    Записывает вопрос, ответ и норму в CSV файл с временной меткой.
    """
    log_dir, log_file = "chat_history", "qa_log.csv"
    os.makedirs(log_dir, exist_ok=True)

    log_path = os.path.join(log_dir, log_file)

    if not os.path.isfile(log_path):
        with open(log_path, mode="w", newline="", encoding="utf-8") as file:
            writer = csv.writer(file)
            writer.writerow(["timestamp", "question", "answer", "norma"])

    # Добавляем запись в журнал
    with open(log_path, mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        writer.writerow([timestamp, question, answer, norma])

