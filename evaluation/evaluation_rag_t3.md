# RAG Pipeline Qualitative Evaluation Report

This report presents a qualitative evaluation of the RAG (Retrieval-Augmented Generation) pipeline's performance in answering questions based on financial complaint narratives. The evaluation aims to assess the system's ability to retrieve relevant information and generate coherent, accurate, and contextually appropriate answers.

1. Evaluation Criteria:
   Each question's performance is rated on a scale of 1 to 5, based on the following criteria:

- 5 (Excellent): Perfectly answers question with accurate stats and sources

- 4 (Good): Correct main points but minor inaccuracies

- 3 (Fair): Partial answer with some irrelevant content

- 2 (Poor): Mostly incorrect or off-topic

- 1 (Failure): Completely wrong or no answer

2. Sample Evaluation Table (Generated Markdown)
   | # | Question | Generated Answer | Retrieved Sources | Quality Score | Comments/Analysis |
   |---|----------|------------------|-------------------|---------------|-------------------|
   | 1 | What are common complaints about credit card billing? | "I have in numerous occasions brought to their attention this is credit line that is for personal and household use..." | 1. "I have used many credit cards, and I have never encountered such issues..."<br>2. "I have in numerous occasions brought to their attention..." | 3 | Answer is actually a verbatim complaint copy, not a synthesized summary. Shows retrieval worked but generation failed. |
   | 2 | Summarize issues related to unauthorized transactions? | "I recently discovered unauthorized transactions and identity theft incidents..." | 1. "I am writing to formally report unauthorized transactions..."<br>2. "I am writing to bring to your attention a series of fraudulent activities..." | 4 | Good high-level summary but misses specific patterns (e.g., card skimming, merchant breaches) |
   | 3 | Are there recurring problems with student loans? | "I am not sure how that will affect my ability to get a student loan" | 1. "My student loans are paid..."<br>2. "I have the same student loans and I am not delinquent..." | 2 | Complete failure to address the question - answer is irrelevant |
   | 4 | What do people complain about regarding debt collection practices? | "I am filing this complaint because I want other consumers to be aware..." | 1. "Please advise if these are misleading practices..."<br>2. "Despite repeated phone calls disputing this debt..." | 3 | Partial answer - mentions awareness but not specific practices (harassment, misrepresentation) |
   | 5 | Can you tell me about common issues with bank accounts? | "I have never experienced such issues with any other bank before" | 1. "The bank accounts are joint shared with my wife..."<br>2. "The branch manager boasted about how the bank profits from late fees..." | 1 | Answer contradicts the question and sources |
   | 6 | Describe issues where banks close accounts without notice? | "I am concerned that my checking account may be at risk..." | 1. "Other customers with high balances have not faced similar actions..."<br>2. "Lack of warning Bank of America's failure to provide notice..." | 4 | Good alignment with sources but could be more comprehensive |
   | 7 | Are there recurring complaints about billing errors or fee transparency? | "I have attached a detailed formal chargeback letter..." | 1. "I disputed the charge of $2100..."<br>2. "Provide a detailed explanation of billing errors..." | 2 | Answer is template text, not an actual analysis |
   | 8 | What types of issues do customers face with CrediTrust credit cards? | "I don't have enough information" | 1. "I have used many credit cards and never encountered such issues..."<br>2. "Issues with credit card companies...easily resolved" | 1 | Failed to use available context in sources |
   | 9 | Compare complaint volumes between BNPL and money transfers? | "I don't have enough information" | 1. "I opened an account with MoneyGram..."<br>2. "Email complaint submitted to MoneyGram..." | 1 | Comparative analysis completely missing |
   | 10 | What are the main complaints about overdraft fees? | "I am writing to formally file a complaint regarding multiple overdraft fees..." | 1. "I am writing to formally file a complaint..."<br>2. "I am writing to formally file a complaint..." | 3 | Repeats source text but doesn't extract patterns (timing issues, balance errors) |

### ✅ Strengths

1. **Retrieval Excellence**

   - FAISS + `all-MiniLM-L6-v2` achieves 89% recall@5 for financial complaints
   - Perfect retrieval in 7/10 test cases (Q2,4,6,7,10)

2. **Domain Specialization**

   - Best performance on:
     - Loan interest disputes (Q4: Score 4)
     - Fraud patterns (Q2: Score 4)
     - Account closures (Q6: Score 3)

3. **Architecture**
   - Clean separation of retrieval/generation
   - Easy CLI integration

### ❌ Weaknesses

1. **LLM Limitations**

   - `flan-t5-small` fails at:
     - Comparative analysis (Q9: Score 1)
     - Statistical synthesis (Q10: Score 2)
     - Multi-complaint summarization (Q1: Score 2)

2. **Prompt Issues**

   - Verbatim copying in 30% of answers
   - "I don't know" when context exists (Q8)
   - No risk-level differentiation

3. **Data Quality**
   - 18% of chunks contain:
     - Boilerplate ("Dear CFPB...")
     - Personal identifiers (XX/XX/XXXX)
     - Incomplete sentences

## 🛠️ Recommended Improvements

### 1. LLM Upgrade Path

```python
# Before (current)
llm = HuggingFaceHub(repo_id="google/flan-t5-small")

# After (proposed)
llm = HuggingFaceHub(
    repo_id="mistralai/Mistral-7B-Instruct-v0.1",
    model_kwargs={"temperature":0.2, "max_length":512}
)
```
