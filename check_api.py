"""
Pre-flight GPT API health check. Run before any experiment.
Exits with code 1 if the API is not returning valid responses.
"""
import json
import sys
import requests

config = json.load(open('model_configs/gpt3.5_config.json'))
api_key = config['api_key_info']['api_keys'][0]

print("Checking GPT API...", flush=True)

try:
    response = requests.post(
        'https://api.openai.com/v1/chat/completions',
        headers={'Authorization': f'Bearer {api_key}', 'Content-Type': 'application/json'},
        json={
            'model': 'gpt-3.5-turbo',
            'messages': [{'role': 'user', 'content': 'Reply with the word OK only.'}],
            'max_tokens': 5
        },
        timeout=30
    )
except Exception as e:
    print(f"API CHECK FAILED: connection error — {e}")
    sys.exit(1)

if response.status_code != 200:
    print(f"API CHECK FAILED: HTTP {response.status_code} — {response.text[:200]}")
    sys.exit(1)

text = response.json()['choices'][0]['message']['content'].strip()
if not text:
    print("API CHECK FAILED: empty response from GPT")
    sys.exit(1)

print(f"API CHECK OK: '{text}'")
