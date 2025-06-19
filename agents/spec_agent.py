# agents/spec_agent.py

import os
import yaml
import json
import openai
from openai import OpenAI


class SpecAgent:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.0):
        """
        Agent to generate pipeline spec from a BCI device YAML description,
        using the new openai-python v1 API.
        """
        # Initialize v1 client; it picks up OPENAI_API_KEY from env
        self.client = OpenAI()
        self.model = model_name
        self.temperature = temperature

    def load_spec(self, yaml_file: str) -> dict:
        with open(yaml_file, "r") as f:
            return yaml.safe_load(f)

    def generate_pipeline_spec(self, spec_data: dict) -> dict:
        # Build messages
        system_msg = {
            "role": "system",
            "content": (
                "You are an expert system designing a BCI signal processing pipeline. "
                "Given a hardware specification, output a JSON dictionary with keys: "
                "signal_shape, preprocessing, features, decoding, sdk, endpoint."
            ),
        }
        user_msg = {
            "role": "user",
            "content": (
                "Hardware spec:\n```json\n"
                + json.dumps(spec_data, indent=2)
                + "\n```\nGenerate pipeline spec JSON."
            ),
        }

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[system_msg, user_msg],
            temperature=self.temperature,
        )
        text = resp.choices[0].message.content

        # Parse JSON safely
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            snippet = text[start : end + 1]
            return json.loads(snippet)

    def run(self, yaml_file: str) -> dict:
        spec = self.load_spec(yaml_file)
        return self.generate_pipeline_spec(spec)

    def run_from_content(self, yaml_content: str) -> dict:
        spec = yaml.safe_load(yaml_content)
        return self.generate_pipeline_spec(spec)
