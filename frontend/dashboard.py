import streamlit as st
import sys, os
# Insert the project root (one level up) into Python’s module search path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from agents.spec_agent import SpecAgent
from agents.code_agent import CodeAgent

st.title("🧠 NeuroForge: BCI Middleware Builder")

uploaded = st.file_uploader("Upload hardware YAML spec", type=["yaml", "yml"])
if uploaded:
    content = uploaded.read().decode()
    agent = SpecAgent(model_name="gpt-4", temperature=0.0)
    spec = agent.run_from_content(content)
    st.subheader("Generated Pipeline Spec")
    st.json(spec)

    # Generate code stubs
    if st.button("Generate Code Stubs"):
        with st.spinner("Generating code..."):
            stubs = CodeAgent().generate_code(spec)
        for layer, code in stubs.items():
            st.markdown(f"#### {layer}.py")
            st.code(code, language="python")