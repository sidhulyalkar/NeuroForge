# pipeline.py

from agents.spec_agent import SpecAgent
from agents.code_agent import CodeAgent

def run_full_pipeline(yaml_path):
    # 1. Spec → dict
    spec_agent = SpecAgent()
    pipeline_spec = spec_agent.run(yaml_path)

    # 2. Spec dict → code stubs
    code_agent = CodeAgent()
    code_modules = code_agent.generate_code(pipeline_spec)

    return pipeline_spec, code_modules

if __name__ == "__main__":
    spec, modules = run_full_pipeline("hardware_profiles/openbci_cyton.yaml")
    print("SPEC:", spec)
    for name, code in modules.items():
        print(f"\n--- {name}.py ---\n{code}")
