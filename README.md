# Filtering-the-Noise: ML for Trustworthy Location Reviews

Online reviews shape how people perceive local businesses and locations. Yet, irrelevant, misleading, or low-quality reviews can distort reputation and mislead users.

This project, developed for a hackathon challenge, leverages Machine Learning (ML) and Natural Language Processing (NLP) to automatically evaluate the quality and relevancy of Google location reviews, aligning them with platform policies.

The goal is to:
- Improve review reliability
- Enhance user trust
- Support fair representation for businesses




## 🚀 Problem Statement  

We design and implement an ML-based system that:

- Gauges review quality → Detects spam, ads, irrelevant rants, and fake experiences.

- Assesses relevancy → Ensures reviews genuinely relate to the business or location.

- Enforces policies → Flags or filters reviews violating rules such as:

    ❌ Policy 1: No advertisements/promotions

    ❌ Policy 2: No irrelevant/off-topic content

    ❌ Policy 3: No fake rants from non-visitors




## 💡 Motivation & Impact

- For Users → Increases trust in reviews, leading to smarter decisions.

- For Businesses → Reduces malicious/irrelevant reviews, ensuring fairer reputation.

- For Platforms → Automates moderation, reducing manual effort while improving credibility.




## ⚙️ Setup
1) Environment
```bash
# Create virtual environment
python -m venv .venv  

# Activate the environment
source .venv/bin/activate     # For Linux/Mac
.venv\Scripts\activate        # For Windows  

# Install dependencies
pip install -r requirements.txt

```

2) Install Ollama

👉 [Download Ollama](https://ollama.com/download)

3) Start Ollama & Pull Models in Command Prompt
```bash
ollama
ollama pull llama3.2
```

