import json
import requests


def get_gpt_response(content, access_token):

    url = "https://gigachat.devices.sberbank.ru/api/v1/chat/completions"

    system_promt = "Ты профессиональный юрист. Ответь на вопрос пользователя используя статью нормативного акта. Ничего не придумывай. Не пиши ничего лишнего. Если не знаешь ответа, то напиши «не знаю»."

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
