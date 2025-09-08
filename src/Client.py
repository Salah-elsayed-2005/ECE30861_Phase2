# src/Client.py
from abc import ABC, abstractmethod
from typing import Any
import requests
from collections import deque
import time

class Client(ABC):
    """
    Abstract client that enforces a rate-limit check before sending requests.
    Subclasses must implement:
      - can_send(): bool
      - _send(...): Any
    """

    @abstractmethod
    def can_send(self) -> bool:
        """
        Return True if a request is allowed right now (respecting rate limit).
        Return False otherwise.
        """
        ...

    @abstractmethod
    def _send(self, *args: Any, **kwargs: Any) -> Any:
        """
        Do the actual request (HTTP, API call, etc.).
        """
        ...

    def request(self, *args: Any, **kwargs: Any) -> Any:
        """
        Public entrypoint: always check can_send() first, then delegate to _send().
        """
        if not self.can_send():
            raise RuntimeError("Rate limit exceeded: request not allowed right now.")
        return self._send(*args, **kwargs)
    

class GrokClient(Client):
    request_history = deque() # Will keep track of requests done accross all instances of this object

    def __init__(self, max_requests: int, token: str, base_url: str = "https://api.groq.com/openai/v1", window_seconds: float = 60.0) -> None:
        super().__init__()
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.token = token
        self.base_url = base_url

    def can_send(self) -> bool:
        """
        Determines whether or not we have hit our limit and number of requests. We keep of track
        of number of requests within the window using the GrokClient.request_history object 
        """
        # Get current time for the window
        now = time.monotonic()
        cutoff = now - self.window_seconds

        # Remove any requests from before the window
        while GrokClient.request_history and GrokClient.request_history[0] <= cutoff:
            GrokClient.request_history.popleft()

        # If we can still make the request, we keep going
        if len(GrokClient.request_history) < self.max_requests:
            GrokClient.request_history.append(now)
            return True
        return False

    def _send(self, method: str, path: str, **kwargs: Any) -> Any:
        """
        Perform an HTTP request to Grok’s API.

        Example:
            client.request("GET", "/models", params={"limit": 5})
        """
        url = f"{self.base_url}{path}"
        headers = {"Authorization": f"Bearer {self.token}"}
        try:
            resp = requests.request(method=method, url=url, headers=headers, timeout=15, **kwargs)
        except requests.RequestException as e:
            raise RuntimeError(f"Grok API request failed: {e}") from e

        if not resp.ok:
            raise RuntimeError(f"Grok API error {resp.status_code}: {resp.text}")

        # Try to parse JSON, else return text
        try:
            return resp.json()
        except ValueError:
            return resp.text
        
    def llm(self, message: str) -> str:
        """
        Runs llama-3.1-8b-instant LLM on input
        """
        completion = self.request(
            "POST",
            "/chat/completions",
            json={
                "model": "llama-3.1-8b-instant",
                "messages": [{"role": "user", "content": "Hello Grok!"}],
            },
        )
        return completion


if __name__ == "__main__":

    grok = GrokClient(max_requests=3, token="Placeholder")
    for _ in range(5):
        print(grok.llm("hello there"))
        print("\n\\n")
