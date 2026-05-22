import requests

OLLAMA_BASE = "http://localhost:11434"


def check_ollama():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=3)
        return r.status_code == 200
    except Exception:
        return False


def list_models():
    try:
        r = requests.get(f"{OLLAMA_BASE}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def ask_ollama(prompt, system="", model="llama3.2"):
    payload = {"model": model, "prompt": prompt, "system": system, "stream": False}
    try:
        r = requests.post(f"{OLLAMA_BASE}/api/generate", json=payload, timeout=180)
        if r.status_code == 200:
            return r.json().get("response", "")
        return f"[Ollama error {r.status_code}]"
    except requests.exceptions.ConnectionError:
        return "[Ollama is not running — start with: ollama serve]"
    except Exception as e:
        return f"[Error: {e}]"
