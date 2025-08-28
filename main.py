from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from vector import retriever

model = OllamaLLM(model = "llama3.2")

template = """
You are an expert in identifying trustworthy and policy-compliant location reviews.

Reference Materials:
{reference}

Review to Evaluate:
{review}

Policies to Enforce:
  1.  No Advertisement: Reviews should not contain promotional content, discount offers, or links.
  2.  No Irrelevant Content: Reviews should focus on the experience at the location, not other matters.
  3.  No Rant Without Visit: Rants or complaints must come from someone who actually visited, not from speculation or hearsay.

Instructions:
  •  First, determine whether the review is Valid or Flagged.
  •  If the review is flagged, identify which single policy it violated the most (choose the strongest violation).
  •  Provide a brief explanation (1 to 2 sentences) justifying your decision.

Output Format:
• Decision: Valid / Flagged
• Primary Violation (if flagged): [Policy Name]
• Explanation: [Short reasoning]

⸻

Example Output:
• Decision: Flagged
• Primary Violation: No Advertisement
• Explanation: The review contains a promotional link to a discount website, which violates the no advertisement policy.
"""

prompt = ChatPromptTemplate.from_template(template)
chain = prompt | model

while True:
    print("\n\n-----------------------")
    user_input = input("Input your review (q to quit): ") # should include location information to evaluate policy 2
    print("\n\n")
    if user_input == "q":
        break
    # reviews = retriever.invoke(user_input)
    result = chain.invoke({"reference": [], "review": user_input}) # update reference: reviews for RAG
    print(result)