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

## Getting Started

Follow the steps below to set up and run the **Radio-Sim Agent**. You can run the pipeline either:

- On **Google Colab** (quick start, recommended for testing), or
- Locally via **CLI** (VS Code, terminal, etc.).

### 1. Clone the Repository

Clone the repository from GitHub:

```bash

!git clone https://github.com/sajjadhussa1n/radio-sim-agent.git
%cd radio-sim-agent

```

### 2. Install Dependencies

Install all the required libraries.

```bash
!pip install -r requirements.txt
```

### 3. Setup Your OpenAI API Key

Setup your OpenAI API key to run the OpenAI Models. Make sure to keep the key secret and do not share it with others. The following script will secretly enter your OpenAI API keys and set it as the environment variable. 

```bash
from getpass import getpass
import os
api_key = getpass("Paste your OpenAI API key (it will be hidden): ")
os.environ["OPENAI_API_KEY"] = api_key
```

### 4. Run the script

```bash
!python main.py
```


---

## Planned Extensions

- **Multi-tool reasoning** for fine-grained EM feature analysis

- **Contextual Q&A** over simulation results (e.g., “Where is signal weakest?”)

- **Interactive web interface** for file upload and visualization

---

## License

This project is licensed under the  
**Creative Commons Attribution–NonCommercial 4.0 International License (CC BY-NC 4.0)**  
See the [LICENSE](LICENSE) file for details.

RadioSim Agent © 2025 Sajjad Hussain (NUST) and Conor Brennan (DCU).
Licensed under CC BY-NC 4.0 — Non-commercial use only.




