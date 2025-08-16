import gradio as gr
import pandas as pd
import plotly.express as px
from rag_pipeline_updated import RAGGenerator
import warnings

# Suppress huggingface_hub warnings
warnings.filterwarnings("ignore", message=".*huggingface_hub.*")

# Initialize RAG system with better error handling
try:
    print("Initializing RAG system...")
    rag_system = RAGGenerator()
except Exception as e:
    print(f"Failed to initialize RAG system: {str(e)}")
    # Fallback to simpler model if available
    try:
        print("Trying smaller model...")
        rag_system = RAGGenerator("google/flan-t5-base")
    except:
        gr.Error("Failed to load any model")
        raise

def generate_response(question):
    try:
        answer, contexts = rag_system.generate(question)
        
        # Create visualization
        if contexts:
            df = pd.DataFrame([{
                'Product': ctx.product,
                'Relevance': ctx.score,
                'Text': ctx.text[:100] + '...'
            } for ctx in contexts])
            
            fig = px.bar(
                df,
                x='Product',
                y='Relevance',
                color='Product',
                title="Complaint Relevance by Product"
            )
            viz_html = fig.to_html(full_html=False)
        else:
            viz_html = "<p>No data for visualization</p>"
        
        # Format sources
        sources_html = "<div style='max-height: 400px; overflow-y: auto;'>"
        for i, ctx in enumerate(contexts[:3], 1):
            sources_html += f"""
            <div style='margin-bottom: 15px; padding: 10px; background: #f8f9fa; border-radius: 5px;'>
                <b>Source {i}</b> | {ctx.product} | Relevance: {ctx.score:.2f}<br>
                {ctx.text[:200]}...
            </div>
            """
        sources_html += "</div>"
        sources_html = sources_html if contexts else "<p>No supporting complaints found</p>"
        
        return answer, viz_html, sources_html
        
    except Exception as e:
        error_msg = f"Error processing request: {str(e)}"
        return error_msg, "", ""

# Example questions
EXAMPLES = [
    "What are common issues with credit cards?",
    "Show complaints about late payments",
    "Analyze BNPL complaints"
]

# Gradio interface
with gr.Blocks(title="Financial Complaint Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # Financial Complaint Analysis
    Ask questions about customer complaints across financial products
    """)
    
    with gr.Row():
        with gr.Column():
            question = gr.Textbox(
                label="Your Question",
                placeholder="e.g., What are the main issues with savings accounts?",
                lines=3
            )
            gr.Examples(
                examples=EXAMPLES,
                inputs=question,
                label="Example Questions"
            )
            submit_btn = gr.Button("Analyze", variant="primary")
            
        with gr.Column():
            with gr.Tabs():
                with gr.Tab("Analysis"):
                    answer_output = gr.Textbox(label="Response", lines=6)
                with gr.Tab("Visualization"):
                    viz_output = gr.HTML()
                with gr.Tab("Sources"):
                    sources_output = gr.HTML()
    
    submit_btn.click(
        fn=generate_response,
        inputs=question,
        outputs=[answer_output, viz_output, sources_output]
    )

if __name__ == "__main__":
    # Simplified launch for local development
    demo.launch()