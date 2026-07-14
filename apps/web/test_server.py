"""
Simple test server for local development without Docker
Run this to test the web app locally before deploying to Docker
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import uvicorn
from pydantic import BaseModel

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Allow all origins
    allow_credentials=True,
    allow_methods=["*"],  # Allow all methods
    allow_headers=["*"],  # Allow all headers
)

class PredictRequest(BaseModel):
    image_bytes: str
    prompt: str
    Type: str = "Latex"  # Default to Latex mode

@app.get("/")
async def root():
    return {"message": "Image to LaTeX API Server", "status": "running"}

@app.get("/health")
async def health():
    return {"status": "healthy"}

@app.post("/predict")
async def predict(request: PredictRequest):
    """
    Mock prediction endpoint for testing
    Supports two modes:
    - Type="Latex": Pix2Text OCR mode
    - Type="Label": Qwen3-VL graph label recognition mode
    """
    # Simulate processing
    image_hex = request.image_bytes
    prompt = request.prompt
    conversion_type = request.Type
    
    print(f"📥 Received request:")
    print(f"  - Image data length: {len(image_hex)} chars")
    print(f"  - Conversion type: {conversion_type}")
    print(f"  - Prompt: {prompt[:100]}...")
    
    # Mock LaTeX response based on type
    if conversion_type == "Label":
        # Graph label recognition (complex expressions)
        mock_latex = r"F_{p+2}^{2} = F_{p+2} \alpha_{1} \ldots \alpha_{p+2} F_{p+2}^{\alpha_{1} \ldots \alpha_{p+2}}"
        print(f"📊 Using Label mode (Qwen3-VL)")
    else:
        # Regular LaTeX OCR
        mock_latex = r"\frac{x^2 + y^2}{z^2} = 1"
        print(f"🔤 Using Latex mode (Pix2Text)")
    
    print(f"📤 Sending response: {mock_latex}")
    
    # Return plain text (same as real server after fix)
    return mock_latex

if __name__ == "__main__":
    print("🚀 Starting test server...")
    print("📍 Server will run at: http://localhost:8080")
    print("🔧 This is a MOCK server for testing the web app")
    print("💡 For real predictions, use the Docker container\n")
    
    uvicorn.run(
        app, 
        host="0.0.0.0", 
        port=8080,
        log_level="info"
    )
