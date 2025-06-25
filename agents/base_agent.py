# agents/base_agent.py
import logging
from openai import OpenAI
from neuralake.core import Catalog

class Agent:
    def __init__(self,
                 catalog: Catalog = None,
                 model_name: str = "gpt-4",
                 temperature: float = 0.0,
                 client: OpenAI = None):
        self.catalog = catalog
        self.model_name = model_name
        self.temperature = temperature
        self.client = client or OpenAI()
        self.logger = logging.getLogger(self.__class__.__name__)
        # e.g., set up retry/backoff, metrics hooks, etc.

    def run(self, *args, **kwargs):
        raise NotImplementedError("Each agent implements its own run()")

    def _chat(self, messages):
        """Wrap LLM calls for retries, logging, function-calls, etc."""
        self.logger.debug("Sending to LLM: %s", messages)
        resp = self.client.chat.completions.create(
            model=self.model_name,
            messages=messages,
            temperature=self.temperature,
        )
        return resp.choices[0].message.content
