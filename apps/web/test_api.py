"""
Automated test script for both Latex and Label modes
Tests the API endpoints with sample images
"""

import requests
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

def create_test_image_hex():
    """Create a simple test image and convert to hex"""
    from PIL import Image, ImageDraw, ImageFont
    import io
    
    # Create white image
    img = Image.new('RGB', (400, 200), color='white')
    draw = ImageDraw.Draw(img)
    
    # Draw some mathematical expression
    draw.text((50, 80), "x² + y² = z²", fill='black')
    
    # Convert to hex
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr.hex()

def test_latex_mode(api_url="http://localhost:8080/predict"):
    """Test Latex mode (Pix2Text OCR)"""
    print("\n" + "="*60)
    print("🔤 Testing LATEX MODE (Pix2Text OCR)")
    print("="*60)
    
    hex_data = create_test_image_hex()
    
    payload = {
        "image_bytes": hex_data,
        "prompt": "",
        "Type": "Latex"
    }
    
    print(f"📤 Sending request to: {api_url}")
    print(f"   Type: Latex")
    print(f"   Image data length: {len(hex_data)} chars")
    
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        
        if response.ok:
            result = response.text.strip().strip('"').strip("{}").strip("'")
            print(f"\n✅ SUCCESS!")
            print(f"📊 Response: {result}")
            print(f"📏 Length: {len(result)} chars")
            return True, result
        else:
            print(f"\n❌ FAILED!")
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.text}")
            return False, None
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return False, None

def test_label_mode(api_url="http://localhost:8080/predict"):
    """Test Label mode (Qwen3-VL)"""
    print("\n" + "="*60)
    print("📊 Testing LABEL MODE (Qwen3-VL)")
    print("="*60)
    
    hex_data = create_test_image_hex()
    
    payload = {
        "image_bytes": hex_data,
        "prompt": "Extract the mathematical expression and convert to LaTeX",
        "Type": "Label"
    }
    
    print(f"📤 Sending request to: {api_url}")
    print(f"   Type: Label")
    print(f"   Prompt: {payload['prompt']}")
    print(f"   Image data length: {len(hex_data)} chars")
    
    try:
        response = requests.post(api_url, json=payload, timeout=10)
        
        if response.ok:
            result = response.text.strip().strip('"').strip("{}").strip("'")
            print(f"\n✅ SUCCESS!")
            print(f"📊 Response: {result}")
            print(f"📏 Length: {len(result)} chars")
            return True, result
        else:
            print(f"\n❌ FAILED!")
            print(f"   Status: {response.status_code}")
            print(f"   Error: {response.text}")
            return False, None
    
    except Exception as e:
        print(f"\n❌ ERROR: {str(e)}")
        return False, None

def test_connection(api_url="http://localhost:8080"):
    """Test API connection"""
    print("\n" + "="*60)
    print("🔗 Testing API Connection")
    print("="*60)
    
    health_url = api_url.replace('/predict', '/health')
    
    print(f"📤 Checking: {health_url}")
    
    try:
        response = requests.get(health_url, timeout=5)
        if response.ok:
            print(f"✅ Server is running!")
            print(f"   Status: {response.status_code}")
            print(f"   Response: {response.text}")
            return True
        else:
            print(f"⚠️ Server responded with: {response.status_code}")
            return True  # Server is up but endpoint may not exist
    except Exception as e:
        print(f"❌ Connection failed: {str(e)}")
        return False

def run_all_tests():
    """Run all tests"""
    print("\n" + "🧪 AUTOMATED TEST SUITE ".center(60, "="))
    print(f"Testing API at: http://localhost:8080/predict")
    print("="*60)
    
    # Test connection
    if not test_connection():
        print("\n⚠️ Server not responding. Make sure test_server.py is running:")
        print("   python test_server.py")
        return
    
    # Test both modes
    latex_ok, latex_result = test_latex_mode()
    label_ok, label_result = test_label_mode()
    
    # Summary
    print("\n" + "="*60)
    print("📋 TEST SUMMARY")
    print("="*60)
    
    print(f"\n✅ Connection Test: PASSED")
    print(f"{'✅' if latex_ok else '❌'} Latex Mode Test: {'PASSED' if latex_ok else 'FAILED'}")
    print(f"{'✅' if label_ok else '❌'} Label Mode Test: {'PASSED' if label_ok else 'FAILED'}")
    
    if latex_ok and label_ok:
        print("\n🎉 ALL TESTS PASSED!")
        print("\n📊 Results Comparison:")
        print(f"   Latex Mode: {latex_result}")
        print(f"   Label Mode: {label_result}")
    else:
        print("\n⚠️ Some tests failed. Check the output above.")
    
    print("\n" + "="*60)
    print("💡 Next Steps:")
    print("   1. Open browser: http://localhost:8501")
    print("   2. Try drawing/uploading images")
    print("   3. Toggle between Latex and Label modes")
    print("   4. Compare results!")
    print("="*60 + "\n")

if __name__ == "__main__":
    run_all_tests()
