import os
import time
import requests
from rag import load_docs, build_index, answer_question

TELEGRAM_TOKEN = os.environ.get('TELEGRAM_BOT_TOKEN')
YANDEX_TOKEN = os.environ.get('YANDEX_TOKEN')
YANDEX_FOLDER_ID = os.environ.get('YANDEX_FOLDER_ID')

if not (TELEGRAM_TOKEN and YANDEX_TOKEN and YANDEX_FOLDER_ID):
    raise RuntimeError('TELEGRAM_BOT_TOKEN, YANDEX_TOKEN and YANDEX_FOLDER_ID must be set')

BASE_URL = f'https://api.telegram.org/bot{TELEGRAM_TOKEN}'


def get_updates(offset=None):
    params = {'timeout': 100}
    if offset:
        params['offset'] = offset
    resp = requests.get(f'{BASE_URL}/getUpdates', params=params, timeout=60)
    resp.raise_for_status()
    return resp.json().get('result', [])


def send_message(chat_id, text):
    data = {'chat_id': chat_id, 'text': text}
    requests.post(f'{BASE_URL}/sendMessage', data=data, timeout=60)


def main():
    docs, _ = load_docs('docs')
    vectorizer, embeddings = build_index(docs)
    last_update = None
    while True:
        for update in get_updates(last_update):
            last_update = update['update_id'] + 1
            message = update.get('message')
            if not message or 'text' not in message:
                continue
            chat_id = message['chat']['id']
            question = message['text']
            answer = answer_question(question, vectorizer, embeddings, docs, YANDEX_TOKEN, YANDEX_FOLDER_ID)
            send_message(chat_id, answer)
        time.sleep(1)


if __name__ == '__main__':
    main()
