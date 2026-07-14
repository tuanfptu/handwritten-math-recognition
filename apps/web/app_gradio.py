"""
Image to LaTeX Web App using Gradio
Alternative Python-based UI with simpler interface
"""

import gradio as gr
import requests
from PIL import Image
import io

# Default settings
DEFAULT_API_URL = "http://localhost:8080/predict"
DEFAULT_PROMPT = "Convert this handwritten mathematical expression to Label Graph format. Only output the Label Graph code without any explanation."

def image_to_hex(image):
    """Convert PIL Image to hex string"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr.hex()

def convert_image_to_latex(image, api_url, prompt, conversion_type):
    """Convert image to LaTeX using API"""
    if image is None:
        return "⚠️ Please provide an image (draw or upload)", "", ""
    
    try:
        # Ensure image is PIL Image
        if not isinstance(image, Image.Image):
            image = Image.fromarray(image)
        
        # Convert to RGB if needed
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        # Convert to hex
        hex_data = image_to_hex(image)
        
        # Send request with Type field
        payload = {
            "image_bytes": hex_data,
            "prompt": prompt,
            "Type": conversion_type
        }
        
        response = requests.post(api_url, json=payload, timeout=30)
        
        if response.ok:
            latex_code = response.text.strip().strip('"').strip("{}").strip("'")
            mode_info = "🔤 **Pix2Text (Latex OCR)**" if conversion_type == "Latex" else "📊 **Qwen3-VL (Label Recognition)**"
            
            # Create graph info for Label mode
            graph_info = ""
            if conversion_type == "Label":
                import re
                num_subscripts = latex_code.count('_')
                num_superscripts = latex_code.count('^')
                num_symbols = latex_code.count('\\')
                
                # Extract labels
                labels = re.findall(r'([A-Z]_\{[^}]+\}|\w+)', latex_code)
                labels_list = [f"{i}. `{label}`" for i, label in enumerate(labels[:8], 1)]
                labels_text = "\n".join(labels_list) if labels_list else "No labels detected"
                
                graph_info = f"""### 📊 Label Graph Analysis

**Structure:**
- Length: {len(latex_code)} chars  
- Subscripts: {num_subscripts}  
- Superscripts: {num_superscripts}  
- Special symbols: {num_symbols}  

**Extracted Labels:**  
{labels_text}
"""
            
            rendered = f"✅ Conversion successful using {mode_info}\n\n{graph_info}\n\n**Rendered:**\n\n$${latex_code}$$"
            
            return latex_code, rendered, graph_info if conversion_type == "Label" else ""
        else:
            return f"❌ Server error: {response.status_code}", response.text, ""
    
    except Exception as e:
        return f"❌ Error: {str(e)}", "", ""

def test_connection(api_url):
    """Test API connection"""
    try:
        health_url = api_url.replace('/predict', '/health')
        response = requests.get(health_url, timeout=5)
        if response.ok:
            return "✅ Connected successfully!"
        else:
            return f"⚠️ Server responded with status: {response.status_code}"
    except Exception as e:
        return f"❌ Connection failed: {str(e)}"

# Create Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(primary_hue="indigo"),
    title="Image to LaTeX Converter"
) as demo:
    
    # Header
    gr.Markdown(
        """
        # 📐 Image to LaTeX Converter
        ### Draw or upload mathematical expressions to convert them to LaTeX
        """
    )
    
    with gr.Row():
        # Left column: Input
        with gr.Column(scale=1):
            gr.Markdown("### 🎨 Input")
            
            # Tabs for draw/upload
            with gr.Tab("✏️ Draw"):
                draw_input = gr.Paint(
                    label="Draw your expression",
                    type="pil",
                    brush=gr.Brush(default_size=3, colors=["#000000"]),
                    canvas_size=(600, 300),
                )
            
            with gr.Tab("📁 Upload"):
                upload_input = gr.Image(
                    label="Upload image",
                    type="pil",
                    sources=["upload", "clipboard"]
                )
            
            # Settings
            with gr.Accordion("⚙️ Settings", open=False):
                api_url_input = gr.Textbox(
                    label="API Server URL",
                    value=DEFAULT_API_URL,
                    placeholder="http://localhost:8000/predict"
                )
                
                conversion_type = gr.Radio(
                    label="Conversion Type",
                    choices=["Latex", "Label"],
                    value="Latex",
                    info="Latex: Pix2Text OCR | Label: Qwen3-VL graph recognition"
                )
                
                prompt_input = gr.Textbox(
                    label="Prompt (for Label mode only)",
                    value=DEFAULT_PROMPT,
                    lines=3,
                    info="Only used when Type=Label"
                )
                
                test_btn = gr.Button("🔗 Test Connection", size="sm")
                connection_status = gr.Textbox(label="Connection Status", interactive=False)
            
            # Convert button
            convert_btn = gr.Button("🚀 Convert to LaTeX", variant="primary", size="lg")
        
        # Right column: Output
        with gr.Column(scale=1):
            gr.Markdown("### 📊 Results")
            
            latex_output = gr.Code(
                label="LaTeX Code",
                language="latex",
                lines=5
            )
            
            rendered_output = gr.Markdown(
                label="Rendered Expression",
                value=""
            )
            
            # Graph info (only shown for Label mode)
            graph_info_output = gr.Markdown(
                label="Graph Analysis",
                value="",
                visible=True
            )
    
    # Examples
    gr.Markdown("### 📚 Example Images")
    gr.Examples(
        examples=[
            ["example_images/equation1.png"] if False else None,  # Add your example images here
        ],
        inputs=upload_input,
        label="Click to load example"
    )
    
    # Event handlers
    def convert_from_draw(image, api_url, prompt, conv_type):
        return convert_image_to_latex(image, api_url, prompt, conv_type)
    
    def convert_from_upload(image, api_url, prompt, conv_type):
        return convert_image_to_latex(image, api_url, prompt, conv_type)
    
    # Connect draw conversion
    convert_btn.click(
        fn=lambda img1, img2, url, prompt, ctype: convert_image_to_latex(
            img1 if img1 is not None else img2, url, prompt, ctype
        ),
        inputs=[draw_input, upload_input, api_url_input, prompt_input, conversion_type],
        outputs=[latex_output, rendered_output, graph_info_output]
    )
    
    # Test connection
    test_btn.click(
        fn=test_connection,
        inputs=[api_url_input],
        outputs=[connection_status]
    )
    
    # Footer
    gr.Markdown(
        """
        ---
        <p style="text-align: center; color: #64748b;">
        Powered by Qwen3-VL Model | Built with ❤️ for CROHME
        </p>
        """
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True
    )
