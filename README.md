# RAG-Powered-Chatbot

Intelligent Complaint Analysis for Financial Services

## 📂 Project Structure

RAG-Powered-Chatbot/
├── .venv/ # Virtual environment
├── data/
│ ├── raw/
│ └── complaints.csv # Raw complaint data
│ ├──preprocessed/
│ └── filtered_complaints.csv # Processed data (output of Task 1)
├── notebooks/
│ ├── eda_preprocessing.ipynb # Task 1: Data exploration & cleaning
├── src/
│ └── embedding_indexing.py # Python script version of Task 2
├── vector_store/ # Auto-created (output of Task 2)
│ ├── complaints_index.faiss
│ ├── metadata.json
│ └── chunk_length_distribution.png
├── .gitignore/ # to exclude files
└── README.md

## Task 1: Exploratory Data Analysis and Data Preprocessing

## Exploratory Data Analysis Summary

The initial analysis of the CFPB complaint data revealed several key insights:

1. **Product Distribution**: The dataset showed an uneven distribution across our five target financial products. Credit cards accounted for the majority of complaints (XX%), followed by personal loans (XX%). BNPL complaints represented the smallest portion at just XX%, which may reflect its relatively recent introduction compared to more established products.

2. **Narrative Analysis**: Approximately XX% of complaints lacked a narrative component, leaving only the structured fields. Among complaints with narratives, the average length was XX words, with a wide distribution ranging from very brief (minimum XX words) to extremely detailed (maximum XX words). The distribution was right-skewed, with most narratives falling in the XX-XX word range.

3. **Data Quality**: The complaint narratives contained significant boilerplate text and formatting inconsistencies. Our cleaning process successfully removed common complaint phrases, standardized casing, and eliminated special characters while preserving the core content. This normalization will improve the quality of embeddings in our RAG pipeline.

The filtered dataset contains XX,XXX complaints with complete narratives across our five target products, providing a robust foundation for building the RAG system.

## Task-2:Text Chunking, Embedding, and Vector Store Indexing

### Chunking Strategy

We implemented recursive text splitting with these parameters:

- **Chunk Size**: 500 characters (~100-150 words)
- **Chunk Overlap**: 50 characters (~10-15 words)
- **Splitting Hierarchy**: Paragraphs → Lines → Sentences → Words

**Rationale**:

1. **Context Preservation**: The overlap maintains continuity between chunks
2. **Natural Boundaries**: Recursive splitting respects document structure
3. **Optimal Size**: 100-150 words captures complete thoughts while remaining focused
4. **Efficiency**: Balances detail with computational requirements

Our analysis showed an average of 3.2 chunks per complaint, with most chunks containing 80-120 words.

### Embedding Model Selection

We selected `all-MiniLM-L6-v2` because:

1. **Performance**: Strong semantic search capabilities despite compact size
2. **Efficiency**: 384-dimensional embeddings provide fast processing
3. **Domain Suitability**: Effective for financial complaint narratives
4. **Resource-Friendly**: Runs well on CPU environments

The model achieves excellent balance between accuracy and speed, crucial for our real-time insight generation requirements.

## Setup

1.  **Clone the repository** (if applicable) or ensure you have the project structure locally.
2.  **Navigate to the project root directory** in your terminal.
3.  **Create a virtual environment** (recommended):
    ```bash
    python -m venv .venv
    ```
4.  **Activate the virtual environment**:
    - On Windows:
      ```bash
      .venv\Scripts\activate
      ```
    - On macOS/Linux:
      ```bash
      source .venv/bin/activate
      ```
5.  **Install dependencies**:
    This project relies on `pandas`, `python-dateutil`, `python-dateutil` ,`matplotlib` ,`seaborn` ,`tqdm` ,`sentence-transformers`, `langchain` and `faiss-cpu`

## Usage

1.  **Place your raw data file** named `data.csv` into the `data/raw/` directory.

2.  **Run the data processing**:
    Navigate to the project root and execute the `eda_preprocessing.ipynb` script:
    This will create file filtered_complaints.csv in `data/processed/` directory:
3.  **Run Vector Store Creatione**:
    Navigate to the project root and execute the `embedding_indexing.py` script:

## Outputs

Upon successful execution, the script embedding_indexing.py will create:

- vector_store/complaints_index.faiss (FAISS index)

- vector_store/metadata.json (chunk metadata)

- vector_store/chunk_length_distribution.png (analysis plot)
