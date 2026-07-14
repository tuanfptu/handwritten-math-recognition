"""
Image to LaTeX Web App using Gradio - Dual Mode Version
Displays results from BOTH Latex and Label modes simultaneously
"""

import gradio as gr
import requests
from PIL import Image
import io
import re
from label_graph_converter import label_to_latex

# Default settings
DEFAULT_API_URL = "http://localhost:8080/predict"
DEFAULT_PROMPT = "Convert this handwritten mathematical expression to LaTeX format. Only output the LaTeX code without any explanation."

def image_to_hex(image):
    """Convert PIL Image to hex string"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr.hex()

def convert_single_mode(image, api_url, prompt, conversion_type):
    """Convert image using a single mode"""
    try:
        hex_data = image_to_hex(image)
        
        payload = {
            "image_bytes": hex_data,
            "prompt": prompt,
            "Type": conversion_type
        }
        
        # Set timeout based on conversion type
        # Label mode (Qwen3-VL) needs more time for complex processing
        timeout = 120 if conversion_type == "Label" else 30
        
        response = requests.post(api_url, json=payload, timeout=timeout)
        
        if response.ok:
            latex_code = response.text.strip()
            latex_code = latex_code.strip('"').strip("{}").strip("'")
            
            # Clean up LaTeX delimiters if present in response
            latex_code = latex_code.replace('\\n$$\\n', '').replace('$$', '').replace('\\n', '')
            latex_code = latex_code.strip()
            
            return latex_code, None
        else:
            return None, f"Server error: {response.status_code} - {response.text}"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def analyze_latex_structure(latex_code, mode_name):
    """Analyze LaTeX structure and generate info"""
    
    info = f"""### {mode_name}

**Output:**
```
{latex_code}
```

**Metrics:**
- Total Length: {len(latex_code)} chars
"""
    return info

def convert_image_dual_mode(image, api_url, prompt):
    """
    Convert image to LaTeX using BOTH modes simultaneously
    Returns: (latex_info, label_info, comparison_info)
    """
    if image is None:
        return "⚠️ Please draw or upload an image", "", ""
    
    # Ensure image is PIL Image and RGB
    if not isinstance(image, Image.Image):
        image = Image.fromarray(image)
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Convert with Latex mode
    latex_code, latex_error = convert_single_mode(image, api_url, "", "Latex")
    
    # Convert with Label mode
    label_code, label_error = convert_single_mode(image, api_url, prompt, "Label")
    
    # Generate outputs
    latex_info = ""
    if latex_error:
        latex_info = f"### ❌ Latex Mode Error\n\n{latex_error}"
    elif latex_code:
        latex_info = analyze_latex_structure(latex_code, "🔤 Pix2Text (Latex Mode)")
        latex_info += f"\n\n**Rendered LaTeX:**\n\n$${latex_code}$$"
    else:
        latex_info = "⚠️ No result from Latex mode"
    
    label_info = ""
    if label_error:
        label_info = f"### ❌ Label Mode Error\n\n{label_error}"
    elif label_code:
        # Convert label graph to LaTeX
        try:
            converted_latex = label_to_latex(label_code)
            label_info = f"""### 📊 Qwen3-VL (Label Graph Mode)

**Label Graph Output:**
```
{label_code}
```

**Converted to LaTeX:**
```latex
{converted_latex}
```

**Rendered LaTeX:**

$${converted_latex}$$

**Format Information:**
- Output format: Label Relation format (e.g., `1 NoRel - Below 3 Right ! Right 1 Right !`)
- Each symbol/label is separated by relation keywords
- Relations: `Right`, `Below`, `Above`, `Sub`, `Sup`, `Inside`, `NoRel`, etc.

**Metrics:**
- Label Graph Length: {len(label_code)} chars
- LaTeX Length: {len(converted_latex)} chars
"""
        except Exception as e:
            label_info = f"""### 📊 Qwen3-VL (Label Graph Mode)

**Label Graph Output:**
```
{label_code}
```

**⚠️ Conversion Error:** {str(e)}

**Metrics:**
- Label Graph Length: {len(label_code)} chars
"""
    else:
        label_info = "⚠️ No result from Label mode"
    
    # Comparison
    comparison_info = "### 🔄 Side-by-Side Comparison\n\n"
    if latex_code and label_code:
        try:
            converted_label = label_to_latex(label_code)
            length_diff = len(label_code) - len(latex_code)
            same_result = latex_code.strip() == converted_label.strip()
            
            comparison_info += f"""
| Metric | Latex Mode | Label Mode (Raw) | Label Mode (Converted) |
|--------|-----------|-----------|-----------|
| **Output Format** | LaTeX Code | Label Graph Relations | LaTeX Code |
| **Length** | {len(latex_code)} chars | {len(label_code)} chars | {len(converted_label)} chars |

**Latex vs Converted Label Match:** {'✅ Yes - Identical output' if same_result else '❌ No - Different outputs'}

---

**🔤 Latex Mode Output:**
```latex
{latex_code}
```

**📊 Label Graph Output:**
```
{label_code}
```

**🔄 Label Graph Converted to LaTeX:**
```latex
{converted_label}
```

**Visual Comparison:**

Latex Mode: $${latex_code}$$

Converted Label: $${converted_label}$$

**Explanation:**
- **Latex Mode**: Direct OCR to LaTeX (fast, simple formulas)
- **Label Mode**: Graph relations → converted to LaTeX (accurate for complex structures)
- **Comparison**: {'Both produce same result!' if same_result else 'Different approaches may yield different results'}
"""
        except Exception as e:
            comparison_info += f"⚠️ Conversion error: {str(e)}\n\nRaw comparison:\n\n**Latex:** {len(latex_code)} chars\n**Label:** {len(label_code)} chars"
    else:
        comparison_info += "⚠️ Cannot compare - one or both conversions failed"
    
    return latex_info, label_info, comparison_info

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
    title="Image to LaTeX - Dual Mode"
) as demo:
    
    # Header
    gr.Markdown(
        """
        # 🔬 Image to LaTeX Converter - Dual Mode Comparison
        ### Compare results from **Pix2Text (Latex)** and **Qwen3-VL (Label)** models simultaneously
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
            with gr.Accordion("⚙️ Settings", open=True):
                api_url_input = gr.Textbox(
                    label="API Server URL",
                    value=DEFAULT_API_URL,
                    placeholder="http://localhost:8080/predict"
                )
                
                prompt_input = gr.Textbox(
                    label="Prompt (for Label mode)",
                    value=DEFAULT_PROMPT,
                    lines=3,
                    info="Instruction for the Qwen3-VL model"
                )
                
                gr.Markdown("**ℹ️ Dual Mode:** Both Latex and Label conversions will run simultaneously")
                
                test_btn = gr.Button("🔗 Test Connection", size="sm")
                connection_status = gr.Textbox(label="Connection Status", interactive=False)
            
            # Convert button
            convert_btn = gr.Button("🚀 Convert with Both Modes", variant="primary", size="lg")
        
        # Right column: Output
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Results from Both Models")
            
            with gr.Tabs():
                with gr.Tab("🔤 Latex Mode (Pix2Text)"):
                    latex_output = gr.Markdown(
                        label="Pix2Text Results",
                        value="*Click 'Convert with Both Modes' to see results*"
                    )
                
                with gr.Tab("📊 Label Mode (Qwen3-VL)"):
                    label_output = gr.Markdown(
                        label="Qwen3-VL Results",
                        value="*Click 'Convert with Both Modes' to see results*"
                    )
                
                with gr.Tab("🔄 Comparison"):
                    comparison_output = gr.Markdown(
                        label="Side-by-Side Comparison",
                        value="*Click 'Convert with Both Modes' to see comparison*"
                    )
    
    # Event handlers
    convert_btn.click(
        fn=lambda img1, img2, url, prompt: convert_image_dual_mode(
            img1 if img1 is not None else img2, url, prompt
        ),
        inputs=[draw_input, upload_input, api_url_input, prompt_input],
        outputs=[latex_output, label_output, comparison_output]
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
        ### 💡 How it works:
        - **🔤 Latex Mode (Pix2Text)**: Fast OCR optimized for mathematical formulas
        - **📊 Label Mode (Qwen3-VL)**: Advanced vision-language model for complex graph labels
        - **🔄 Comparison**: Detailed side-by-side analysis of both results
        
        ### 🎯 Use cases:
        - Compare accuracy of different models
        - Verify complex mathematical expressions
        - Choose the best result for your needs
        - Educational: Learn how different models interpret the same image
        
        <p style="text-align: center; color: #64748b; margin-top: 20px;">
        Powered by Qwen3-VL & Pix2Text | Built with ❤️ using Gradio
        </p>
        """
    )

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7861,  # Different port to avoid conflict
        share=False,
        show_error=True
    )
