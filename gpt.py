import json
import requests
from utils import load_config

secrets = load_config("config/secrets.yaml")


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


def get_gpt_response(content, access_token):

    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    system_promt = "Ты русскоязычный Юрист. Ответь на вопрос пользователя используя только статью. Ничего не придумывай. Не пиши ничего лишнего. Если ответа нет в статье, то напиши «не знаю»."

    payload = json.dumps(
        {
            "model": "GigaChat",
            "messages": [
                {"role": "system", "content": system_promt},
                {"role": "user", "content": content},
            ],
        }
    )

    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
        "Authorization": f"Bearer {access_token}",
    }

    response = requests.request(
        "POST", url, headers=headers, data=payload, verify="./russiantrustedca.pem"
    )
    return json.loads(response.text)["choices"][0]["message"]["content"]
