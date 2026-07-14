"""
Image to LaTeX Web App using Streamlit
Simple Python-based UI for converting handwritten math to LaTeX
"""

import streamlit as st
import requests
from PIL import Image
import io
import base64
from streamlit_drawable_canvas import st_canvas
from label_graph_converter import label_to_latex

# Page configuration
st.set_page_config(
    page_title="Image to LaTeX Converter",
    page_icon="📐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main-header {
        font-size: 3rem;
        color: #4f46e5;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .subtitle {
        text-align: center;
        color: #64748b;
        margin-bottom: 2rem;
    }
    .latex-output {
        background-color: #1e293b;
        color: #e2e8f0;
        padding: 1rem;
        border-radius: 0.5rem;
        font-family: monospace;
        font-size: 1.1rem;
    }
    .rendered-latex {
        background-color: #f8fafc;
        padding: 2rem;
        border-radius: 0.5rem;
        border: 1px solid #e2e8f0;
        font-size: 1.5rem;
        text-align: center;
        min-height: 100px;
    }
    </style>
""", unsafe_allow_html=True)

# Header
st.markdown('<h1 class="main-header">📐 Image to LaTeX Converter</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">Draw or upload mathematical expressions to convert them to LaTeX</p>', unsafe_allow_html=True)

# Sidebar settings
with st.sidebar:
    st.header("⚙️ Settings")
    
    api_url = st.text_input(
        "API Server URL",
        value="https://8000-01k9kneq7qchd1wbgcy1698mf9.cloudspaces.litng.ai/predict",
        help="URL of the prediction API endpoint",
        key="api_url_input"
    )
    
    # Show current API URL
    st.caption(f"🌐 Current API: `{api_url}`")
    
    # Dual mode processing
    st.info("🔄 **Dual Mode**: Both Latex and Label modes will be processed simultaneously for comparison")
    
    prompt_label = st.text_area(
        "Prompt (for Label mode)",
        value="Convert this handwritten mathematical expression to Graph label format. Only output the LaTeX code without any explanation.",
        height=100,
        help="Instruction for the Label mode (Qwen3-VL)"
    )
    
    st.divider()
    
    # Test connection
    if st.button("🔗 Test Connection", use_container_width=True):
        try:
            health_url = api_url.replace('/predict', '/health')
            response = requests.get(health_url, timeout=5)
            if response.ok:
                st.success("✅ Connected successfully!")
            else:
                st.warning(f"⚠️ Server responded with status: {response.status_code}")
        except Exception as e:
            st.error(f"❌ Connection failed: {str(e)}")
    
    st.divider()
    st.info("💡 **Tip**: Use the test server for quick testing without Docker")
    st.code("python test_server.py", language="bash")

# Main content
tab1, tab2 = st.tabs(["✏️ Draw Expression", "📁 Upload Image"])

# Helper function to convert image to hex
def image_to_hex(image):
    """Convert PIL Image to hex string"""
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr.hex()

# Helper function to send request
def convert_to_latex(image, api_url, prompt, conversion_type):
    """Send image to API and get LaTeX response"""
    try:
        # Convert image to hex
        hex_data = image_to_hex(image)
        
        # Send request
        payload = {
            "image_bytes": hex_data,
            "prompt": prompt,
            "Type": conversion_type  # "Latex" or "Label"
        }
        
        # Set timeout based on conversion type
        # Label mode (Qwen3-VL) needs more time for complex processing
        timeout = 120 if conversion_type == "Label" else 30
        
        response = requests.post(api_url, json=payload, timeout=timeout)
        
        if response.ok:
            latex_code = response.text.strip()
            # Remove any quotes or braces if present
            latex_code = latex_code.strip('"').strip("{}").strip("'")
            
            # Clean up LaTeX delimiters if present in response
            # Remove $$, \n, and other formatting artifacts
            latex_code = latex_code.replace('\\n$$\\n', '').replace('$$', '').replace('\\n', '')
            latex_code = latex_code.strip()
            
            return latex_code, None
        else:
            return None, f"Server error: {response.status_code} - {response.text}"
    
    except Exception as e:
        return None, f"Error: {str(e)}"

def convert_both_modes(image, api_url, prompt_label):
    """Convert using both Latex and Label modes"""
    results = {}
    
    # Convert with Latex mode
    with st.spinner("🔤 Converting with Pix2Text (Latex mode)..."):
        latex_result, latex_error = convert_to_latex(image, api_url, "", "Latex")
        results['latex'] = {'code': latex_result, 'error': latex_error}
    
    # Convert with Label mode
    with st.spinner("📊 Converting with Qwen3-VL (Label mode)..."):
        label_result, label_error = convert_to_latex(image, api_url, prompt_label, "Label")
        results['label'] = {'code': label_result, 'error': label_error}
    
    return results

# Tab 1: Draw Expression
with tab1:
    st.subheader("Draw your mathematical expression")
    
    col1, col2 = st.columns([2, 1])
    
    with col2:
        stroke_width = st.slider("Pen Size", 1, 20, 3)
        stroke_color = st.color_picker("Pen Color", "#000000")
        drawing_mode = st.selectbox(
            "Drawing Mode",
            ["freedraw", "line", "rect", "circle"],
            index=0
        )
    
    with col1:
        # Canvas for drawing
        canvas_result = st_canvas(
            fill_color="rgba(255, 255, 255, 0)",
            stroke_width=stroke_width,
            stroke_color=stroke_color,
            background_color="#FFFFFF",
            height=300,
            width=700,
            drawing_mode=drawing_mode,
            key="canvas",
        )
    
    col_btn1, col_btn2 = st.columns(2)
    
    with col_btn1:
        convert_draw = st.button("🚀 Convert to LaTeX", key="convert_draw", use_container_width=True, type="primary")
    
    with col_btn2:
        if st.button("🗑️ Clear Canvas", key="clear_canvas", use_container_width=True):
            st.rerun()
    
    # Process drawing
    if convert_draw:
        if canvas_result.image_data is not None:
            # Convert canvas to PIL Image (fix deprecation warning)
            import numpy as np
            img_array = canvas_result.image_data.astype('uint8')
            img = Image.fromarray(img_array, mode='RGBA')
            # Convert RGBA to RGB
            img_rgb = Image.new('RGB', img.size, (255, 255, 255))
            img_rgb.paste(img, mask=img.split()[3])
            
            # Convert with both modes
            results = convert_both_modes(img_rgb, api_url, prompt_label)
            
            if results:
                st.session_state['results'] = results
        else:
            st.warning("⚠️ Please draw something on the canvas first")

# Tab 2: Upload Image
with tab2:
    st.subheader("Upload an image of a mathematical expression")
    
    uploaded_file = st.file_uploader(
        "Choose an image file",
        type=['png', 'jpg', 'jpeg'],
        help="Supported formats: PNG, JPG, JPEG"
    )
    
    if uploaded_file is not None:
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.image(image, caption="Uploaded Image", use_container_width=True)
        
        with col2:
            st.write("**Image Information:**")
            st.write(f"- Format: {image.format}")
            st.write(f"- Size: {image.size[0]} x {image.size[1]} px")
            st.write(f"- Mode: {image.mode}")
        
        if st.button("🚀 Convert to LaTeX", key="convert_upload", use_container_width=True, type="primary"):
            # Convert to RGB if needed
            if image.mode != 'RGB':
                image = image.convert('RGB')
            
            # Convert with both modes
            results = convert_both_modes(image, api_url, prompt_label)
            
            if results:
                st.session_state['results'] = results

# Display results
st.divider()

if 'results' in st.session_state and st.session_state['results']:
    st.header("📊 Results - Dual Mode Comparison")
    
    results = st.session_state['results']
    
    # Create tabs for each mode
    tab_latex, tab_label, tab_compare = st.tabs(["🔤 Latex Mode", "📊 Label Mode", "🔄 Side-by-Side Comparison"])
    
    # Tab 1: Latex Mode Results
    with tab_latex:
        st.subheader("🔤 Latex OCR Results")
        
        if results['latex']['error']:
            st.error(f"❌ {results['latex']['error']}")
        elif results['latex']['code']:
            latex_code = results['latex']['code']
            
            col1, col2 = st.columns([1, 1])
            
            with col1:
                st.write("**LaTeX Code:**")
                st.code(latex_code, language="latex")
                
                if st.button("📋 Copy Latex Code", key="copy_latex"):
                    st.success("✅ Select and copy the code above!")
            
            with col2:
                st.write("**Rendered:**")
                try:
                    st.markdown(f"$$\n{latex_code}\n$$")
                except:
                    st.warning("⚠️ Could not render LaTeX")
                
                st.metric("Length", f"{len(latex_code)} chars")
        else:
            st.info("No result from Latex mode")
    
    # Tab 2: Label Mode Results
    with tab_label:
        st.subheader("📊 Qwen3-VL Label Recognition Results")
        
        if results['label']['error']:
            st.error(f"❌ {results['label']['error']}")
        elif results['label']['code']:
            label_code = results['label']['code']
            
            # Display Label Graph as text (format: label1 Relation label2 Relation label3...)
            st.write("### 📊 Label Graph (Relation Format)")
            st.info("**Label Graph Output:**")
            st.code(label_code, language="text")
            
            if st.button("📋 Copy Label Graph", key="copy_label"):
                st.success("✅ Select and copy the code above!")
            
            st.write("---")
            
            # Convert Label Graph to LaTeX
            try:
                st.write("### 🔄 Converted to LaTeX")
                converted_latex = label_to_latex(label_code)
                
                col1, col2 = st.columns([1, 1])
                
                with col1:
                    st.write("**LaTeX Code:**")
                    st.code(converted_latex, language="latex")
                    
                    if st.button("📋 Copy Converted LaTeX", key="copy_converted"):
                        st.success("✅ Select and copy the code above!")
                
                with col2:
                    st.write("**Rendered:**")
                    try:
                        st.markdown(f"$$\n{converted_latex}\n$$")
                    except:
                        st.warning("⚠️ Could not render LaTeX")
                    
                    st.metric("Length", f"{len(converted_latex)} chars")
                
                st.write("---")
            except Exception as e:
                st.error(f"❌ Conversion error: {str(e)}")
            
            st.write("**Example format:** `1 NoRel - Below 3 Right ! Right 1 Right !`")
            st.write("- Each symbol/label is separated by relation keywords")
            st.write("- Relations: `Right`, `Below`, `Above`, `Sub`, `Sup`, `Inside`, `NoRel`, etc.")
        else:
            st.info("No result from Label mode")
    
    # Tab 3: Side-by-Side Comparison
    with tab_compare:
        st.subheader("🔄 Direct Comparison")

        col1, col2 = st.columns(2)
        
        with col1:
            st.write("### 🔤 Latex Mode")
            st.info("**Model:** Pix2Text OCR")
            
            if results['latex']['code']:
                latex_code = results['latex']['code']
                st.code(latex_code, language="latex")
                st.metric("Length", f"{len(latex_code)} chars")
                
                try:
                    st.markdown(f"**Rendered:**\n\n$$\n{latex_code}\n$$")
                except:
                    st.warning("⚠️ Could not render")
            else:
                st.error("❌ No result")
        
        with col2:
            st.write("### 📊 Label Mode")
            st.info("**Model:** Qwen3-VL (Label Graph)")
            
            if results['label']['code']:
                label_code = results['label']['code']
                
                st.write("**Label Graph Output:**")
                st.code(label_code, language="text")
                
                # Convert to LaTeX
                try:
                    converted_latex = label_to_latex(label_code)
                    st.write("**Converted LaTeX:**")
                    st.code(converted_latex, language="latex")
                    st.metric("Length", f"{len(converted_latex)} chars")
                    
                    st.write("**Rendered:**")
                    try:
                        st.markdown(f"$$\n{converted_latex}\n$$")
                    except:
                        st.warning("⚠️ Could not render")
                except Exception as e:
                    st.error(f"Conversion error: {str(e)}")
            else:
                st.error("❌ No result")
        
        # Comparison metrics
        if results['latex']['code'] and results['label']['code']:
            st.divider()
            st.write("### 📈 Comparison Metrics")
            
            col_m1, col_m2 = st.columns(2)
            
            with col_m1:
                length_diff = len(results['label']['code']) - len(results['latex']['code'])
                st.metric(
                    "Length Difference",
                    f"{abs(length_diff)} chars",
                    f"{'Label' if length_diff > 0 else 'Latex'} is longer"
                )
            
            with col_m2:
                same_result = results['latex']['code'].strip() == results['label']['code'].strip()
                st.metric(
                    "Output Format",
                    "LaTeX vs Label Graph",
                    "Different formats" if not same_result else "Same"
                )
    
    # Clear results button
    st.divider()
    if st.button("🔄 Clear Results and Start Over"):
        del st.session_state['results']
        st.rerun()

# Footer
st.divider()
st.markdown(
    '<p style="text-align: center; color: #64748b;">Powered by Qwen3-VL Model | Built with ❤️ for CROHME</p>',
    unsafe_allow_html=True
)
