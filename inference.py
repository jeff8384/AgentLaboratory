import json
import requests
import tiktoken

TOKENS_IN = {}
TOKENS_OUT = {}

try:
    encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    encoding = None


def local_llama_generate(messages, temperature=None, max_new_tokens=512):
    payload = {
        "model": "llama-3.1-8b",
        "messages": messages,
        "stream": False,
        "options": {"num_predict": max_new_tokens},
    }
    if temperature is not None:
        payload["options"]["temperature"] = temperature
    response = requests.post("http://localhost:11434/api/chat", json=payload)
    response.raise_for_status()
    return response.json()["message"]["content"].strip()


def curr_cost_est():
    costmap_in = {"llama-3.1-8b": 0.20 / 1000000}
    costmap_out = {"llama-3.1-8b": 0.80 / 1000000}
    return sum(
        costmap_in[m] * TOKENS_IN.get(m, 0) for m in TOKENS_IN
    ) + sum(costmap_out[m] * TOKENS_OUT.get(m, 0) for m in TOKENS_OUT)


def query_model(model_str, prompt, system_prompt, temp=None, **_):
    if model_str not in [
        "llama-3.1-8b",
        "llama3.1-8b",
        "Llama 3.1:8B",
        "llama-3.1-8B",
    ]:
        raise ValueError("Only llama-3.1-8b is supported")
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    answer = local_llama_generate(messages, temperature=temp)
    if encoding is not None:
        TOKENS_IN[model_str] = TOKENS_IN.get(model_str, 0) + len(
            encoding.encode(system_prompt + prompt)
        )
        TOKENS_OUT[model_str] = TOKENS_OUT.get(model_str, 0) + len(
            encoding.encode(answer)
        )
    return answer

