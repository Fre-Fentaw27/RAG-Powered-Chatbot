# ---------- IMPORTS ----------
import gradio as gr
from rag_pipeline_t3 import RAGGenerator, extract_key_phrases

# Initialize the RAG system
rag_system = RAGGenerator()

def generate_response(question):
    """Handle question submission and format the response with sources"""
    # Generate answer and retrieve contexts
    answer, contexts = rag_system.generate(question)
    
    # Prepare sources display
    sources_html = "<h3>Sources Used:</h3>"
    for i, context in enumerate(contexts[:3], 1):  # Show top 3 sources
        sources_html += f"""
        <div style='margin-bottom: 15px; padding: 10px; background: #f5f5f5; border-radius: 5px;'>
            <b>Source {i}:</b> (Product: {context['product']}, Score: {context['score']:.2f})<br>
            {context['text']}
        </div>
        """
    
    return answer, sources_html

# Define the Gradio interface
with gr.Blocks(title="CrediTrust Complaint Analyzer", theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 🔍 CrediTrust Complaint InsightBot")
    gr.Markdown("Ask about specific customer service issues to get insights from complaints database.")
    
    with gr.Row():
        with gr.Column():
            question_input = gr.Textbox(
                label="Your Question",
                placeholder="Type your question about customer complaints here...",
                lines=3
            )
            submit_btn = gr.Button("Ask", variant="primary")
            clear_btn = gr.Button("Clear")
        
        with gr.Column():
            answer_output = gr.Textbox(
                label="Generated Answer",
                interactive=False,
                lines=5
            )
            sources_output = gr.HTML(
                label="Sources Used",
                visible=True
            )
    
    # Event handlers
    submit_btn.click(
        fn=generate_response,
        inputs=question_input,
        outputs=[answer_output, sources_output]
    )
    
    clear_btn.click(
        fn=lambda: ["", ""],
        outputs=[question_input, answer_output]
    )

if __name__ == "__main__":
    demo.launch()