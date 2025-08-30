# POLARIS: Policy-Aligned Local Review Inference System

Online reviews shape how people perceive local businesses and locations. Yet, irrelevant, misleading or low-quality reviews can distort reputation and mislead users.

This project, developed for a hackathon challenge, leverages Machine Learning (ML) and Natural Language Processing (NLP) to automatically evaluate the quality and relevancy of online location reviews, aligning them with platform policies.

The goal is to:
- Improve review reliability
- Enhance user trust
- Support fair representation for businesses




## 🚀 Problem Statement  

We design and implement an ML-based system that:

- Gauges review quality: Detects spam, ads, irrelevant rants and fake experiences.

- Assesses relevancy: Ensures reviews genuinely relate to the business or location.

- Enforces policies: Flags or filters reviews violating rules such as

    ❌ Policy 1: No advertisements/promotions

    ❌ Policy 2: No irrelevant/off-topic content

    ❌ Policy 3: No fake rants from non-visitors




## 💡 Motivation & Impact

- For Users: Increases trust in reviews, leading to smarter decisions.

- For Businesses: Reduces malicious/irrelevant reviews, ensuring fairer reputation.

- For Platforms: Automates moderation, reducing manual effort while improving credibility.




## ⚙️ Setup
1) Requirements
- Create a file named requirements.txt with the following:
```bash
langchain
langchain-ollama
langchain-chroma
pandas
```

2) Environment
```bash
# Create virtual environment
python -m venv .venv  

# Activate the environment
source .venv/bin/activate     # For Linux/Mac
.venv\Scripts\activate        # For Windows  

# Install dependencies
pip install -r requirements.txt

```

3) Install Ollama

[Download Ollama here](https://ollama.com/download)

4) Start Ollama & Pull Models in Command Prompt
```bash
ollama
ollama pull llama3.2
```

## ▶️ Usage
### Evaluate Reviews (Main Script)

The [main.py](https://github.com/edwin-ljx/Filtering-the-Noise-ML-for-Trustworthy-Location-Reviews/blob/main/main.py) script is interactive and supports two modes:
  
- Individual review input (manual typing in terminal)

```
# Example

Mode: (i) individual review, (f) CSV file, (q) quit: i
Enter the location being reviewed (q to quit): Pizza House
Enter your review: Best pizza! Visit www.pizzapromo.com for discounts!

Evaluating...

Decision: Flagged
Primary Violation: No Advertisement
Explanation: The review contains the website www.pizzapromo.com and mentions "Visit for discounts!", which promotes a specific URL and discount offer, violating the policy of no advertisement.
```

          
- CSV file batch evaluation (eg: [location_reviews.csv](https://github.com/edwin-ljx/Filtering-the-Noise-ML-for-Trustworthy-Location-Reviews/blob/main/location_reviews.csv))

```
# Example

Mode: (i) individual review, (f) CSV file, (q) quit: f
Path to CSV file: /Users/scormon/Downloads/location_reviews.csv

Evaluating CSV rows...

Processed row 1: Decision=Flagged
Processed row 2: Decision=Valid
Processed row 3: Decision=Valid
Processed row 4: Decision=Flagged
Processed row 5: Decision=Valid
Processed row 6: Decision=Flagged
Processed row 7: Decision=Flagged
Processed row 8: Decision=Flagged
Processed row 9: Decision=Valid
Processed row 10: Decision=Valid

Done. Output written to: /Users/scormon/Downloads/location_reviews_evaluated.csv
```
   
### Test with Ground Truth (Test Script)

The [test.py](https://github.com/edwin-ljx/Filtering-the-Noise-ML-for-Trustworthy-Location-Reviews/blob/main/test.py) script helps compare predicted outputs (eg: [location_reviews_evaluated.csv](https://github.com/edwin-ljx/Filtering-the-Noise-ML-for-Trustworthy-Location-Reviews/blob/main/location_reviews_evaluated.csv)) against ground-truth labels and generates accuracy + [mismatch reports](https://github.com/edwin-ljx/Filtering-the-Noise-ML-for-Trustworthy-Location-Reviews/blob/main/mismatches.csv).

```
# Example

=== Review Evaluation Script ===

Enter path to CSV file: /Users/scormon/Downloads/location_reviews_evaluated.csv

CSV headers detected: ['review_id', 'location', 'review', 'correct_labels', 'correct_decision', 'Decision', 'Primary Violation', 'Explanation']
Prediction Decision column [Decision]: Decision
Prediction Violation column [Primary Violation]: Primary Violation
Ground Truth Decision column [GT_Decision]: correct_decision
Ground Truth Violation column [GT_Violation]: correct_labels
ID column (optional, Enter to skip): review_id

=== Results ===
Total rows: 10
Decision Accuracy: 9/10 = 90.00%
Violation Accuracy: 7/10 = 70.00%

Saved mismatch report: mismatches.csv
```

## 🕹️ Demo
Try out an interactive demo of the project by downloading and running [demo.ipynb](https://github.com/edwin-ljx/Filtering-the-Noise-ML-for-Trustworthy-Location-Reviews/blob/main/demo.ipynb)!
