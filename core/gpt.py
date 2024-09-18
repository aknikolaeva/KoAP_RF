import json
import requests
from constants import TEMPERATURE, MAX_TOKENS, REPETITION_PENALTY, SYSTEM_PROMT


def get_gpt_response(content, access_token):

    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    payload = json.dumps(
        {
            "model": "GigaChat",
            "messages": [
                {"role": "system", "content": SYSTEM_PROMT},
                {"role": "user", "content": content},
            ],
            "temperature": TEMPERATURE,
            "max_tokens": MAX_TOKENS,
            "repetition_penalty": REPETITION_PENALTY,
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
