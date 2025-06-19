import json
from agents.spec_agent import SpecAgent

if __name__ == "__main__":
    # Default to OpenBCI Cyton spec; change or add CLI parsing as needed
    yaml_file = "hardware_profiles/openbci_cyton.yaml"

    agent = SpecAgent(model_name="gpt-4", temperature=0.0)
    pipeline_spec = agent.run(yaml_file)

    print("=== Generated Pipeline Specification ===")
    print(json.dumps(pipeline_spec, indent=2))
