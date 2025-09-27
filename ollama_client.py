import requests
import json

class OllamaClient:
    """
    Minimal wrapper around the Ollama /api/generate endpoint
    for local inference.
    """

    def __init__(self, model="mistral", host="http://localhost:11434"):
        """
        model: e.g. 'llama2:7b', 'mistral', etc.
        host:  The server base URL, e.g. 'http://localhost:11434'
        """
        self.model = model
        self.host = host

    def text_generation(self, prompt, max_new_tokens=250, temperature=0.7):
        """
        Send a request to /api/generate and stream back the entire response.
        """
        data = {
            "model": self.model,
            "prompt": prompt,
            "temperature": temperature,
            "numCtx": 2048,
            "max_tokens": max_new_tokens
        }

        # IMPORTANT: in Ollama 0.6.x, the path must be /api/generate, not /generate
        url = f"{self.host}/api/generate"
        response = requests.post(url, json=data, stream=True)
        full_text = ""

        for line in response.iter_lines(decode_unicode=True):
            if not line:
                continue
            try:
                chunk = json.loads(line)
                full_text += chunk.get("response", "")
            except json.JSONDecodeError:
                pass

        return full_text.strip() 