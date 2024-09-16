from constants import EMBEDDING_MODEL_NAME

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from typing import List, Tuple

# Функция для пулинга скрытых состояний
def pool(hidden_state, mask, pooling_method="cls"):
    if pooling_method == "mean":
        s = torch.sum(hidden_state * mask.unsqueeze(-1).float(), dim=1)
        d = mask.sum(axis=1, keepdim=True).float()
        return s / d
    elif pooling_method == "cls":
        return hidden_state[:, 0]

class EmbeddingGenerator:
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_NAME)
        self.model = AutoModel.from_pretrained(EMBEDDING_MODEL_NAME)

    def token_chunker(self, context, max_chunk_size=512, stride=128, min_chunk_len=50):
        # Токенизируем контекст один раз.
        # Если контекст > max_chunk_size, разбиваем его на несколько чанков с перекрытием stride

        context_tokens = self.tokenizer.encode(context, add_special_tokens=False)

        chunk_holder = []
        chunk_size = max_chunk_size
        current_pos = 0
        while current_pos < len(context_tokens):
            end_point = (
                current_pos + chunk_size
                if (current_pos + chunk_size) < len(context_tokens)
                else len(context_tokens)
            )
            token_chunk = context_tokens[current_pos:end_point]

            # Пропускаем чанки, которые короче min_chunk_len
            if len(token_chunk) < min_chunk_len:
                current_pos = end_point
                continue

            # Создаем маску внимания для каждого токена
            attention_mask = torch.ones((1, len(token_chunk)), dtype=torch.int32)

            # Преобразуем чанк токенов в тензор
            token_chunk = torch.tensor(token_chunk, dtype=torch.int32).unsqueeze(0)

            chunk_holder.append(
                {
                    "token_ids": token_chunk,
                    "context": self.tokenizer.decode(
                        context_tokens[current_pos:end_point], skip_special_tokens=True
                    ),
                    "attention_mask": attention_mask,
                }
            )
            current_pos = current_pos + chunk_size - stride

        return chunk_holder

    # Метод для получения эмбеддингов
    def get_embeddings(
        self, context: str, max_length: int = 512, overlap: int = 128, min_chunk_len: int = 10
    ) -> Tuple[List[str], List[List[float]]]:
        chunks = self.token_chunker(context, max_chunk_size=max_length, stride=overlap, min_chunk_len=min_chunk_len)

        embeddings = []
        documents = []
        for chunk in chunks:
            input_ids = chunk["token_ids"]
            attention_mask = chunk["attention_mask"]

            with torch.no_grad():
                outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

            chunk_embeddings = pool(
                outputs.last_hidden_state,
                attention_mask,
                pooling_method="cls",  # или попробуйте "mean"
            )
            embeddings.append(chunk_embeddings)

            documents.append(chunk["context"])

        # Проверка, что список embeddings не пуст
        if not embeddings:
            return [], []  # Возвращаем пустые списки, если нет валидных чанков

        # Объединяем эмбеддинги из всех чанков
        embeddings = torch.cat(embeddings, dim=0)

        embeddings_list = embeddings.tolist()

        return documents, embeddings_list