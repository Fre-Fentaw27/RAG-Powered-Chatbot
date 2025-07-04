# RAG-Powered-Chatbot

Intelligent Complaint Analysis for Financial Services

## Exploratory Data Analysis Summary

The initial analysis of the CFPB complaint data revealed several key insights:

1. **Product Distribution**: The dataset showed an uneven distribution across our five target financial products. Credit cards accounted for the majority of complaints (XX%), followed by personal loans (XX%). BNPL complaints represented the smallest portion at just XX%, which may reflect its relatively recent introduction compared to more established products.

2. **Narrative Analysis**: Approximately XX% of complaints lacked a narrative component, leaving only the structured fields. Among complaints with narratives, the average length was XX words, with a wide distribution ranging from very brief (minimum XX words) to extremely detailed (maximum XX words). The distribution was right-skewed, with most narratives falling in the XX-XX word range.

3. **Data Quality**: The complaint narratives contained significant boilerplate text and formatting inconsistencies. Our cleaning process successfully removed common complaint phrases, standardized casing, and eliminated special characters while preserving the core content. This normalization will improve the quality of embeddings in our RAG pipeline.

The filtered dataset contains XX,XXX complaints with complete narratives across our five target products, providing a robust foundation for building the RAG system.
