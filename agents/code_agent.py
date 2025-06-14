# agents/code_agent.py

import json
import openai
from openai import OpenAI

class CodeAgent:
    def __init__(self, model_name: str = "gpt-4", temperature: float = 0.0):
        """
        Agent to generate Python code stubs from a pipeline spec dict.
        """
        self.client = OpenAI()
        self.model = model_name
        self.temperature = temperature

    def generate_code(self, pipeline_spec: dict) -> dict:
        """
        For each key in pipeline_spec (e.g. 'preprocessing', 'features', etc.),
        ask the LLM to generate a Python stub. Returns a dict:
            { layer_name: code_string, ... }
        """
        stubs = {}
        for layer, spec in pipeline_spec.items():
            prompt = (
                f"# {layer} module stub\n"
                f"# Based on spec: {json.dumps(spec)}\n\n"
                "def run(data):\n"
                "    \"\"\"\n"
                f"    {layer.capitalize()} step for BCI middleware.\n"
                "    Input: data (NumPy array or DataFrame)\n"
                "    Output: processed data\n"
                "    \"\"\"\n"
                "    # TODO: implement this\n"
                "    pass\n"
            )
            resp = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": "You are a Python expert."},
                    {"role": "user", "content": prompt},
                ],
                temperature=self.temperature,
            )
            code = resp.choices[0].message.content
            stubs[layer] = code
        return stubs
