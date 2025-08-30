from langchain_ollama.llms import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
import pandas as pd
# from vector import retriever

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


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
  df = df.drop_duplicates()
  df = df.dropna()
  df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
  df['rating'] = df['rating'].clip(lower=0, upper=5)
  df = df.dropna(subset=['rating'])
  df['rating'] = df['rating'].astype(int)
  
  df = df[df['text'].astype(str).str.strip() != ""]
  
  return df 

def read_file(file_name: str) -> pd.DataFrame:
  file_format = file_name.split(".")[-1].lower()

  if file_format == "csv":
    df = pd.read_csv(file_name)
  elif file_format in ["xls", "xlsx", "xlsm", "xlsb", "ods"]:
    df = pd.read_excel(file_name)
  elif file_format == "json":
    df = pd.read_json(file_name)
  elif file_format == "xml":
    df = pd.read_xml(file_name)
  else:
    raise ValueError(f"Unsupported file format: {file_format}")
  
  return df
  
while True:
  user_input = ''
  print("\n\n")
  mode = input("Do you want to check reviews individually or from a file? (type 'individual', 'file', or 'quit'): ").strip().lower()

  if mode == "quit":
      print("Exiting program. Goodbye!")
      
  elif mode == "individual":
      review = input("Enter your review (or type 'quit' to exit): ").strip()
      user_input = review
      if review.lower() == "quit":
          print("Exiting program. Goodbye!")
          break

  elif mode == "file":
      file_name = input("Enter the file name (with extension, e.g. data.csv) or type 'quit' to exit: ").strip()
      if file_name.lower() == "quit":
          print("Exiting program. Goodbye!")
          break
      else: 
        clean_df = clean_dataframe(read_file(file_name))
        if len(clean_df['text']) > 0: 
          # get top 5 of the comment instead of all, can change later
          user_input = clean_df['text'].head(5)

  print("\n\n")
  # reviews = retriever.invoke(user_input)
  result = chain.invoke({"reference": [], "review": user_input}) # update reference: reviews for RAG
  print(result)

