
# agents/code_agent.py

from agents.base_agent import Agent
import json

class CodeAgent(Agent):
    def __init__(self,
                 pipeline_catalog=None,
                 model_name: str = "gpt-4",
                 temperature: float = 0.0,
                 client=None):
        super().__init__(catalog=pipeline_catalog,
                         model_name=model_name,
                         temperature=temperature,
                         client=client)
        """
        Agent to generate Python code stubs from a pipeline spec dict.
        """
    def generate_code(self, pipeline_spec: dict) -> dict:
        """
        For each key in pipeline_spec (e.g. 'preprocessing', 'features', etc.),
        ask the LLM to generate a Python stub. Returns a dict:
            { layer_name: code_string, ... }
        """
        stubs = {}
        for layer, spec in pipeline_spec.items():
            # 1) Optionally fetch metadata about inputs/outputs from Neuralake:
            table_schema = {}
            if self.catalog and "tables" in spec:
                tbl = self.catalog.db("bci").table(spec["tables"])
                table_schema = tbl.schema.to_dict()
            # 2) Build a prompt:
            prompt = (
                f"# {layer} module stub\n"
                f"# Based on spec: {json.dumps(spec)}\n"
                f"# Table schema: {json.dumps(table_schema)}\n\n"
                "def run(data):\n"
                '    """\n'
                f"    {layer.capitalize()} step for BCI middleware.\n"
                "    Input: data (NumPy array or DataFrame)\n"
                "    Output: processed data\n"
                '    """\n'
                "    # TODO: implement this\n"
                "    pass\n"
            )
            resp = self._chat([
                {"role": "system", "content": "You are a Python expert specialized in BCI workflows."},
                {"role": "user", "content": prompt},
            ])
            code = resp
            stubs[layer] = code
        return stubs
