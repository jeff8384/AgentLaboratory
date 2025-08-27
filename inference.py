import openai
import time, tiktoken
from openai import OpenAI
import os, anthropic, json, requests
import google.generativeai as genai
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

TOKENS_IN = dict()
TOKENS_OUT = dict()

try:
    encoding = tiktoken.get_encoding("cl100k_base")
except Exception:
    encoding = None

_LOCAL_LLAMA_MODEL = None
_LOCAL_LLAMA_TOKENIZER = None


def load_local_llama_model(model_path=None):
    model_path = model_path or os.getenv("LLAMA_31_8B_PATH", "models/llama-3.1-8b-instruct")
    global _LOCAL_LLAMA_MODEL, _LOCAL_LLAMA_TOKENIZER
    if _LOCAL_LLAMA_MODEL is None or _LOCAL_LLAMA_TOKENIZER is None:
        # Verify model path exists and contains essential files before loading
        if not os.path.isdir(model_path):
            raise FileNotFoundError(
                f"Local LLaMA model directory '{model_path}' not found. "
                "Download the model locally and set LLAMA_31_8B_PATH to the directory containing the model.\n"
                "Expected structure: <model_path>/config.json, tokenizer.model, tokenizer_config.json, ..."
            )
        config_path = os.path.join(model_path, "config.json")
        if not os.path.isfile(config_path):
            raise FileNotFoundError(
                f"Expected config.json inside '{model_path}' but none was found. "
                "Ensure the model directory contains config.json and tokenizer files."
            )

        _LOCAL_LLAMA_TOKENIZER = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
        _LOCAL_LLAMA_MODEL = AutoModelForCausalLM.from_pretrained(model_path, local_files_only=True)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        _LOCAL_LLAMA_MODEL.to(device).eval()
    return _LOCAL_LLAMA_MODEL, _LOCAL_LLAMA_TOKENIZER


def ollama_chat(messages, temperature=None, max_new_tokens=512):
    base_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")
    model_name = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
    payload = {"model": model_name, "messages": messages, "stream": False}
    options = {}
    if temperature is not None:
        options["temperature"] = temperature
    if max_new_tokens is not None:
        options["num_predict"] = max_new_tokens
    if options:
        payload["options"] = options
    resp = requests.post(f"{base_url}/api/chat", json=payload, timeout=60)
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def local_llama_generate(messages, temperature=None, max_new_tokens=512):
    use_ollama = os.getenv("USE_OLLAMA", "").lower() in ["1", "true", "yes"]
    if not use_ollama:
        try:
            model, tokenizer = load_local_llama_model()
            device = model.device
            inputs = tokenizer.apply_chat_template(messages, return_tensors="pt").to(device)
            gen_kwargs = {"max_new_tokens": max_new_tokens}
            if temperature is not None and temperature > 0:
                gen_kwargs.update({"do_sample": True, "temperature": temperature})
            with torch.no_grad():
                outputs = model.generate(inputs, **gen_kwargs)
            return tokenizer.decode(outputs[0, inputs.shape[-1]:], skip_special_tokens=True).strip()
        except FileNotFoundError:
            use_ollama = True
    if use_ollama:
        return ollama_chat(messages, temperature=temperature, max_new_tokens=max_new_tokens)

def curr_cost_est():
    costmap_in = {
        "llama-3.1-8b": 0.20 / 1000000,
        "gpt-4o": 2.50 / 1000000,
        "gpt-4o-mini": 0.150 / 1000000,
        "o1-preview": 15.00 / 1000000,
        "o1-mini": 3.00 / 1000000,
        "claude-3-5-sonnet": 3.00 / 1000000,
        "deepseek-chat": 1.00 / 1000000,
        "o1": 15.00 / 1000000,
        "o3-mini": 1.10 / 1000000,
    }
    costmap_out = {
        "llama-3.1-8b": 0.80 / 1000000,
        "gpt-4o": 10.00/ 1000000,
        "gpt-4o-mini": 0.6 / 1000000,
        "o1-preview": 60.00 / 1000000,
        "o1-mini": 12.00 / 1000000,
        "claude-3-5-sonnet": 12.00 / 1000000,
        "deepseek-chat": 5.00 / 1000000,
        "o1": 60.00 / 1000000,
        "o3-mini": 4.40 / 1000000,
    }
    return sum([costmap_in[_]*TOKENS_IN[_] for _ in TOKENS_IN]) + sum([costmap_out[_]*TOKENS_OUT[_] for _ in TOKENS_OUT])

def query_model(model_str, prompt, system_prompt, openai_api_key=None, gemini_api_key=None,  anthropic_api_key=None, tries=5, timeout=5.0, temp=None, print_cost=True, version="1.5"):
    preloaded_api = os.getenv('OPENAI_API_KEY')
    groq_api = os.getenv('GROQ_API_KEY')
    if openai_api_key is None and preloaded_api is not None:
        openai_api_key = preloaded_api
    if openai_api_key is None and anthropic_api_key is None and groq_api is None:
        if model_str not in ["llama-3.1-8b", "llama3.1-8b", "Llama 3.1:8B", "llama-3.1-8B"]:
            raise Exception("No API key provided in query_model function")
    if openai_api_key is not None:
        openai.api_key = openai_api_key
        os.environ["OPENAI_API_KEY"] = openai_api_key
    if anthropic_api_key is not None:
        os.environ["ANTHROPIC_API_KEY"] = anthropic_api_key
    if gemini_api_key is not None:
        os.environ["GEMINI_API_KEY"] = gemini_api_key
    for _ in range(tries):
        try:
            if model_str in ["llama-3.1-8b", "llama3.1-8b", "Llama 3.1:8B", "llama-3.1-8B"]:
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                llama_api_key = openai_api_key or groq_api
                if llama_api_key:
                    client = OpenAI(api_key=llama_api_key, base_url=os.getenv("GROQ_API_BASE", "https://api.groq.com/openai/v1"))
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="llama-3.1-8b-instruct", messages=messages,)
                    else:
                        completion = client.chat.completions.create(
                            model="llama-3.1-8b-instruct", messages=messages, temperature=temp)
                    answer = completion.choices[0].message.content
                else:
                    answer = local_llama_generate(messages, temperature=temp)

            elif model_str == "gpt-4o-mini" or model_str == "gpt4omini" or model_str == "gpt-4omini" or model_str == "gpt4o-mini":
                model_str = "gpt-4o-mini"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp
                        )
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-mini-2024-07-18", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content

            elif model_str == "gemini-2.0-pro":
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(model_name="gemini-2.0-pro-exp-02-05", system_instruction=system_prompt)
                answer = model.generate_content(prompt).text
            elif model_str == "gemini-1.5-pro":
                genai.configure(api_key=gemini_api_key)
                model = genai.GenerativeModel(model_name="gemini-1.5-pro", system_instruction=system_prompt)
                answer = model.generate_content(prompt).text
            elif model_str == "o3-mini":
                model_str = "o3-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o3-mini-2025-01-31", messages=messages)
                answer = completion.choices[0].message.content

            elif model_str == "claude-3.5-sonnet":
                client = anthropic.Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
                message = client.messages.create(
                    model="claude-3-5-sonnet-latest",
                    system=system_prompt,
                    messages=[{"role": "user", "content": prompt}])
                answer = json.loads(message.to_json())["content"][0]["text"]
            elif model_str == "gpt4o" or model_str == "gpt-4o":
                model_str = "gpt-4o"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    if temp is None:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages
                        )
                    else:
                        completion = openai.ChatCompletion.create(
                            model=f"{model_str}",  # engine = "deployment_name".
                            messages=messages, temperature=temp)
                else:
                    client = OpenAI()
                    if temp is None:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, )
                    else:
                        completion = client.chat.completions.create(
                            model="gpt-4o-2024-08-06", messages=messages, temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "deepseek-chat":
                model_str = "deepseek-chat"
                messages = [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": prompt}]
                if version == "0.28":
                    raise Exception("Please upgrade your OpenAI version to use DeepSeek client")
                else:
                    deepseek_client = OpenAI(
                        api_key=os.getenv('DEEPSEEK_API_KEY'),
                        base_url="https://api.deepseek.com/v1"
                    )
                    if temp is None:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages)
                    else:
                        completion = deepseek_client.chat.completions.create(
                            model="deepseek-chat",
                            messages=messages,
                            temperature=temp)
                answer = completion.choices[0].message.content
            elif model_str == "o1-mini":
                model_str = "o1-mini"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-mini-2024-09-12", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1":
                model_str = "o1"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model="o1-2024-12-17",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-2024-12-17", messages=messages)
                answer = completion.choices[0].message.content
            elif model_str == "o1-preview":
                model_str = "o1-preview"
                messages = [
                    {"role": "user", "content": system_prompt + prompt}]
                if version == "0.28":
                    completion = openai.ChatCompletion.create(
                        model=f"{model_str}",  # engine = "deployment_name".
                        messages=messages)
                else:
                    client = OpenAI()
                    completion = client.chat.completions.create(
                        model="o1-preview", messages=messages)
                answer = completion.choices[0].message.content

            try:
                if model_str in ["o1-preview", "o1-mini", "claude-3.5-sonnet", "o1", "o3-mini"]:
                    encoding = tiktoken.encoding_for_model("gpt-4o")
                elif model_str in ["deepseek-chat", "llama-3.1-8b"]:
                    encoding = tiktoken.encoding_for_model("cl100k_base")
                else:
                    encoding = tiktoken.encoding_for_model(model_str)
                if model_str not in TOKENS_IN:
                    TOKENS_IN[model_str] = 0
                    TOKENS_OUT[model_str] = 0
                TOKENS_IN[model_str] += len(encoding.encode(system_prompt + prompt))
                TOKENS_OUT[model_str] += len(encoding.encode(answer))
                if print_cost:
                    print(f"Current experiment cost = ${curr_cost_est()}, ** Approximate values, may not reflect true cost")
            except Exception as e:
                if print_cost: print(f"Cost approximation has an error? {e}")
            return answer
        except Exception as e:
            print("Inference Exception:", e)
            time.sleep(timeout)
            continue
    raise Exception("Max retries: timeout")


#print(query_model(model_str="o1-mini", prompt="hi", system_prompt="hey"))