# RadioSim Agent

**RadioSim Agent: Combining Large Language Models and Deterministic EM Simulators for Interactive Radio Map Analysis**  
_Agentic AI framework for radio propagation simulation and intelligent query-driven interaction with EM-based channel models._

---

## Overview

**RadioSim Agent** is an **LLM-integrated simulation assistant** that connects **deterministic electromagnetic (EM) ray-tracing models** with **large language models (LLMs)** to enable **interactive, natural language-driven radio channel analysis**.

This system allows researchers and engineers to:
- Run **pathloss and E-field simulations** using deterministic EM solvers.
- **Upload building/environment data** to construct simulation maps.
- Query the system in **natural language** — e.g.,  
  > “Simulate radio signal strength for a UAV transmitter at (320, 470, 25)”  
- Visualize intermediate results (LOS, reflections, ground effects, etc.).
- Conduct **interactive Q&A** with the simulation outputs.

The RadioSim Agent combines:
- **LLMs** (via OpenAI’s GPT-4o-mini)
- **Custom physics-based tools** (LOS, reflection, ground reflection, and pathloss models)
- **LangChain ReAct agent orchestration**


---

## Planned Extensions

- **Multi-tool reasoning** for fine-grained EM feature analysis

- **Contextual Q&A** over simulation results (e.g., “Where is signal weakest?”)

- **Interactive web interface** for file upload and visualization


