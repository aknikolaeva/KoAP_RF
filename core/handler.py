import json
import requests
import time

from core.utils import load_config

secrets = load_config("config/secrets.yaml")


def get_access_token():
    """
    Получает токен доступа от API.
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


class AccessTokenManager:
    def __init__(self):
        self.access_token_cache = {"access_token": None, "expires_at": None}

    def check_cached_access_token(self):
        """
        Проверяет кэшированный токен доступа и обновляет его, если он истек.
        """
        current_time = int(time.time()) * 1000
        if (
            self.access_token_cache["expires_at"] is None
        ) or current_time >= self.access_token_cache["expires_at"]:
            self.access_token_cache = get_access_token()

    def get_access_token(self):
        """
        Возвращает текущий токен доступа.
        """
        self.check_cached_access_token()
        return self.access_token_cache["access_token"]
