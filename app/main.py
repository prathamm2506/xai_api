"""
FastAPI application for Lung Cancer Classification API.
"""
# 1️⃣ Disable CUDA / GPU Initialization (before importing TensorFlow)
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import io
import base64
import httpx
from dotenv import load_dotenv

# Load environment variables from .env file
# Look for .env in the parent directory (ui folder)
env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
load_dotenv(env_path)

from .gradcam import generate_visualizations
from datetime import datetime
from typing import Optional

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from contextlib import asynccontextmanager

from .model_loader import model_loader
from .preprocessing import (
    load_and_preprocess_image,
    load_image_for_visualization,
    IMG_SIZE
)
from .gradcam import (
    make_gradcam_heatmap,
    apply_lung_mask,
    create_overlay,
    get_attention_regions
)
from PIL import Image
import numpy as np


# Initialize FastAPI app with modern lifespan
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model on startup."""
    try:
        model_loader.load_model()
        print("Model loaded successfully.")
    except Exception as e:
        print(f"Startup model load failed: {e}")
    yield

app = FastAPI(
    title="Lung Cancer Classification API",
    description="API for lung cancer detection using DenseNet121 with Grad-CAM explainability",
    version="1.0.0",
    lifespan=lifespan
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Pydantic models for responses
class HealthResponse(BaseModel):
    status: str
    message: str
    model_loaded: bool
    timestamp: str


class ClassResponse(BaseModel):
    classes: dict
    count: int


class PredictionSimpleResponse(BaseModel):
    success: bool
    classification: str
    confidence: float
    all_probabilities: dict
    timestamp: str


class PredictionFullResponse(BaseModel):
    success: bool
    classification: str
    confidence: float
    classes: list
    all_probabilities: dict
    images: dict
    regions: list
    analysis: dict
    recommendations: list
    gemini_report: str
    timestamp: str


class GeminiReportRequest(BaseModel):
    classification: str
    confidence: float
    classes: list
    analysis: dict
    regions: list


class GeminiReportResponse(BaseModel):
    success: bool
    report: str
    timestamp: str




@app.get("/", response_model=HealthResponse)
async def root():
    """
    Health check endpoint.
    """
    model_loaded = model_loader._model is not None
    
    return HealthResponse(
        status="healthy" if model_loaded else "degraded",
        message="Lung Cancer Classification API is running",
        model_loaded=model_loaded,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/health")
async def health_check():
    """
    Detailed health check.
    """
    model_loaded = model_loader._model is not None
    
    return {
        "status": "healthy" if model_loaded else "degraded",
        "model_loaded": model_loaded,
        "model_path": "densenet_focal_phase3.keras",
        "classes": model_loader.get_classes() if model_loaded else [],
        "input_size": IMG_SIZE,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.get("/classes", response_model=ClassResponse)
async def get_classes():
    """
    Get available classification classes.
    """
    classes = model_loader.get_classes()
    class_indices = model_loader.get_class_indices()
    
    return ClassResponse(
        classes=class_indices,
        count=len(classes)
    )


@app.post("/predict/simple", response_model=PredictionSimpleResponse)
async def predict_simple(file: UploadFile = File(...)):
    """
    Simple prediction endpoint - returns classification without visualizations.
    Faster than full prediction.
    """
    # Check if model is loaded
    if model_loader._model is None:
        try:
            model_loader.load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")
    
    try:
        # 5️⃣ File Validation & Security Checks
        # Read and validate file
        contents = await file.read()
        
        # File Size Limit (10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large")
        
        # Content Type Validation
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # Read and preprocess image
        image = Image.open(io.BytesIO(contents))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        img_array = load_and_preprocess_image(image)
        
        # Get predictions
        model = model_loader.get_model()
        # 2️⃣ Fix Keras Input Structure Warning - use named input
        predictions = model.predict({"input_layer": img_array}, verbose=0)
        
        # Handle different output formats from model.predict
        # Some models return a tuple (output, metadata) or just the output array
        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]
        
        # Ensure predictions is a numpy array
        predictions = np.array(predictions)
        
        # If still a batch (2D), take the first sample
        if len(predictions.shape) > 1:
            predictions = predictions[0]
        
        # Flatten if still multi-dimensional
        while len(predictions.shape) > 1:
            predictions = predictions.flatten()
        
        # Convert to Python list for safe indexing
        predictions = predictions.tolist()
        
        # Get class with highest probability
        predicted_class_idx = int(np.argmax(predictions))
        classification = model_loader.INDICES_TO_CLASS[predicted_class_idx]
        confidence = float(np.max(predictions))
        
        # Build probability dictionary
        all_probabilities = {
            model_loader.INDICES_TO_CLASS[i]: float(predictions[i])
            for i in range(len(predictions))
        }
        
        return PredictionSimpleResponse(
            success=True,
            classification=classification,
            confidence=confidence * 100,  # Convert to percentage
            all_probabilities=all_probabilities,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict", response_model=PredictionFullResponse)
async def predict(file: UploadFile = File(...)):
    """
    Full prediction endpoint with Grad-CAM visualizations.
    Returns classification, confidence, heatmap, and explanations.
    """

    # Ensure model is loaded
    if model_loader._model is None:
        try:
            model_loader.load_model()
        except Exception as e:
            raise HTTPException(status_code=503, detail=f"Model not available: {str(e)}")

    try:
        # ----------------------------
        # 5️⃣ File Validation & Security Checks
        # ----------------------------
        contents = await file.read()
        
        # File Size Limit (10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="Image too large")
        
        # Content Type Validation
        if file.content_type not in ["image/jpeg", "image/png"]:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        # ----------------------------
        # 1️⃣ Read Image
        # ----------------------------
        image = Image.open(io.BytesIO(contents))

        if image.mode != "RGB":
            image = image.convert("RGB")

        # Resize once for consistent visualization
        original_array = np.array(image, dtype=np.uint8)

        # ----------------------------
        # 2️⃣ Preprocess for Model
        # ----------------------------
        img_array = load_and_preprocess_image(image)

        model = model_loader.get_model()

        # ----------------------------
        # 3️⃣ Prediction
        # ----------------------------
        predictions = model.predict({"input_layer": img_array}, verbose=0)

        if isinstance(predictions, (tuple, list)):
            predictions = predictions[0]

        predictions = np.array(predictions)

        if len(predictions.shape) > 1:
            predictions = predictions[0]

        predictions = predictions.flatten()

        predicted_class_idx = int(np.argmax(predictions))
        classification = model_loader.INDICES_TO_CLASS[predicted_class_idx]
        confidence = float(np.max(predictions)) * 100

        # Sorted class list
        classes = [
            {
                "name": model_loader.INDICES_TO_CLASS[i],
                "probability": float(predictions[i]) * 100
            }
            for i in range(len(predictions))
        ]
        classes.sort(key=lambda x: x["probability"], reverse=True)

        # Probability dictionary
        all_probabilities = {
            model_loader.INDICES_TO_CLASS[i]: float(predictions[i]) * 100
            for i in range(len(predictions))
        }

        # ----------------------------
        # 4️⃣ Generate Visualizations (Same as Colab)
        # ----------------------------
        layer_name = model_loader.get_last_conv_layer()

        visuals = generate_visualizations(
            np.array(image),   # no resize
            model,
            layer_name
        )


        # visuals contains:
        # original, heatmap, heatmap_array, gradcam, overlay, predicted_class, confidence

        images = {
            "original": visuals["original"],
            "heatmap": visuals["heatmap"],
            "gradcam": visuals["gradcam"],
            "overlay": visuals["overlay"]
        }

        # ----------------------------
        # 4️⃣ Avoid Base64 Re-Decoding (Memory Optimization)
        # ----------------------------
        # Use heatmap_array directly instead of re-decoding from base64
        heatmap_np = visuals["heatmap_array"]

        regions = get_attention_regions(heatmap_np, threshold=0.4)

        # ----------------------------
        # 6️⃣ Analysis + Recommendations
        # ----------------------------
        analysis = generate_analysis_text(
            classification,
            confidence,
            len(regions)
        )

        recommendations = generate_recommendations(classification)

        # ----------------------------
        # 7️⃣ Generate Gemini Report (Dynamic)
        # ----------------------------
        # Prepare data for Gemini with all analysis results
        gemini_data = {
            "classification": classification,
            "confidence": confidence,
            "classes": classes,
            "analysis": analysis,
            "regions": regions,
            "all_probabilities": all_probabilities,
            "recommendations": recommendations
        }
        
        # Automatically generate Gemini report
        gemini_report = await generate_gemini_report(gemini_data)

        # ----------------------------
        # 8️⃣ Final Response
        # ----------------------------
        return PredictionFullResponse(
            success=True,
            classification=classification,
            confidence=confidence,
            classes=classes,
            all_probabilities=all_probabilities,
            images=images,
            regions=regions,
            analysis=analysis,
            recommendations=recommendations,
            gemini_report=gemini_report,
            timestamp=datetime.utcnow().isoformat()
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


def generate_analysis_text(classification: str, confidence: float, region_count: int) -> dict:
    """Generate analysis text based on classification."""
    
    if classification == "Normal":
        summary = f"The AI analysis indicates normal lung tissue with {confidence:.1f}% confidence. No significant abnormalities were detected in the submitted CT scan image."
        details = f"The deep learning model examined the lung field and identified {region_count} region(s) for detailed analysis. The tissue density, nodule characteristics, and overall lung architecture appear within normal parameters."
    elif classification == "Benign":
        summary = f"The AI analysis suggests benign (non-cancerous) findings with {confidence:.1f}% confidence. The detected abnormalities appear consistent with benign conditions."
        details = f"The model identified {region_count} region(s) of interest showing characteristics associated with benign lesions. These findings do not indicate malignancy but should be monitored as per clinical guidelines."
    else:  # Malignant
        summary = f"The AI analysis indicates malignant (cancerous) findings with {confidence:.1f}% confidence. This classification requires immediate clinical correlation and further diagnostic workup."
        details = f"The convolutional neural network identified {region_count} region(s) of interest showing features associated with malignant tumors. The attention mapping indicates concentrated activation in specific areas that warrant urgent medical attention."
    
    return {
        "summary": summary,
        "details": details
    }


def generate_recommendations(classification: str) -> list:
    """Generate recommendations based on classification."""
    
    if classification == "Normal":
        return [
            "Continue routine screening as recommended by healthcare provider",
            "Maintain healthy lifestyle habits including smoking cessation if applicable",
            "Report any new respiratory symptoms to your healthcare provider",
            "Schedule follow-up imaging as per standard screening guidelines"
        ]
    elif classification == "Benign":
        return [
            "Follow up with your healthcare provider within 3-6 months",
            "Continue regular screening schedule as recommended",
            "Monitor for any new symptoms and report to your doctor",
            "Discuss lifestyle modifications to reduce risk factors"
        ]
    else:  # Malignant
        return [
            "Immediate consultation with a pulmonologist or oncologist is strongly recommended",
            "Additional diagnostic procedures (biopsy, PET scan) may be warranted",
            "Complete clinical history and physical examination should be performed",
            "Discuss treatment options and staging procedures with medical oncology team",
            "Consider genetic testing for targeted therapy eligibility",
            "Multidisciplinary tumor board review is recommended for treatment planning"
        ]


# ============================================
# Gemini API Integration for Layman's Report
# ============================================

async def generate_gemini_report(analysis_data: dict) -> str:
    """
    Generate a comprehensive layman's report using Gemini API.
    The report is dynamically generated based on the actual analysis results.
    Provides detailed, easy-to-understand explanations with numerical figures.
    """
    # Get API key from environment
    api_key = os.environ.get("GEMINI_API_KEY", "")
    
    if not api_key or api_key == "your_gemini_api_key_here":
        return generate_fallback_report(analysis_data)
    
    # Extract data for the prompt
    classification = analysis_data.get("classification", "Unknown")
    confidence = analysis_data.get("confidence", 0)
    classes = analysis_data.get("classes", [])
    analysis = analysis_data.get("analysis", {})
    regions = analysis_data.get("regions", [])
    all_probabilities = analysis_data.get("all_probabilities", {})
    recommendations = analysis_data.get("recommendations", [])
    
    # Build classification distribution text with percentages
    class_dist_text = "\n".join([
        f"- {cls['name']}: {cls['probability']:.1f}%"
        for cls in classes
    ])
    
    # Build all probabilities text
    all_prob_text = "\n".join([
        f"- {name}: {prob:.1f}%"
        for name, prob in all_probabilities.items()
    ])
    
    # Count regions and get details
    region_count = len(regions)
    region_details = ""
    if regions:
        for i, r in enumerate(regions, 1):
            region_details += f"\n  Region {i}: Location ({r.get('x', 0):.1f}, {r.get('y', 0):.1f}), Intensity: {r.get('intensity', 0)*100:.0f}%, Size: {r.get('radius', 0):.1f}"
    
    # Build recommendations text
    rec_text = "\n".join([f"- {rec}" for rec in recommendations])
    
    # Create comprehensive prompt for Gemini
    prompt = f"""You are an expert medical report writer specializing in explaining AI-powered lung CT scan analysis results to patients with NO medical background.

Your task is to create a COMPREHENSIVE, DETAILED, and EASY-TO-UNDERSTAND report that explains the following analysis results:

**ACTUAL ANALYSIS RESULTS FROM THE AI MODEL:**

PRIMARY DIAGNOSIS:
- Classification: {classification}
- Confidence Level: {confidence:.1f}%

DETAILED PROBABILITY BREAKDOWN:
{class_dist_text}

ALL CONDITIONS CONSIDERED:
{all_prob_text}

ANALYSIS SUMMARY FROM AI:
{analysis.get('summary', 'N/A')}

DETAILED FINDINGS:
{analysis.get('details', 'N/A')}

REGIONS OF INTEREST:
- Number of areas found: {region_count}
{region_details}

RECOMMENDATIONS:
{rec_text}

YOUR TASK: Write a detailed report for a worried patient

Write a comprehensive report that includes:

1. Warm Welcome & Reassurance: Start with a compassionate greeting

2. What Happened: Explain what the CT scan showed in simple terms

3. Understanding the Numbers: Explain what {confidence:.1f}% confidence means in everyday terms

4. Detailed Breakdown of Each Condition: For EACH condition, explain what it is in simple terms and what the percentage means

5. Regions of Interest Explained: Explain what these areas mean and what intensity percentages indicate

6. What This Means for You: Clear guidance on next steps

7. Compassionate Closing: Remind them to discuss with their doctor

IMPORTANT:
- Use simple, everyday language
- Use REAL numbers and percentages from the data
- Be thorough and detailed
- Structure with clear headings using ##
- End with encouragement to consult healthcare professionals"""

    # Call Gemini API
    try:
        url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
        
        headers = {
            "Content-Type": "application/json"
        }
        
        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 2048,
                "topP": 0.95,
                "topK": 40
            }
        }
        
        # 3️⃣ Convert Gemini API Call to Async - Use httpx
        # 7️⃣ Reduce Gemini Timeout from 60 to 30
        async with httpx.AsyncClient(timeout=30) as client:
            response = await client.post(url, headers=headers, json=payload)
        
        if response.status_code == 200:
            result = response.json()
            if "candidates" in result and len(result["candidates"]) > 0:
                text = result["candidates"][0]["content"]["parts"][0]["text"]
                return text
            else:
                return generate_fallback_report(analysis_data)
        else:
            print(f"Gemini API error: {response.status_code} - {response.text}")
            return generate_fallback_report(analysis_data)
            
    except Exception as e:
        print(f"Error calling Gemini API: {str(e)}")
        return generate_fallback_report(analysis_data)


def generate_fallback_report(analysis_data: dict) -> str:
    """
    Generate a simple fallback report if Gemini API is not available.
    """
    classification = analysis_data.get("classification", "Unknown")
    confidence = analysis_data.get("confidence", 0)
    classes = analysis_data.get("classes", [])
    
    # Get top 2 class probabilities
    top_classes = sorted(classes, key=lambda x: x["probability"], reverse=True)[:2]
    
    if classification == "Normal":
        return f"""## Your Lung CT Scan Results - Easy Understanding

### What the AI Found:
The analysis shows your lungs appear **NORMAL** with **{confidence:.1f}%** confidence. This means the AI found no signs of cancer or concerning abnormalities in your scan.

### Understanding the Numbers:
Think of it like this: if we had 100 similar scans, the AI is **{confidence:.1f}%** sure this one is normal. That's a very high level of confidence!

The breakdown of what the AI detected:
- Normal tissue: {top_classes[0]['probability']:.1f}%
{f"- Other conditions: {top_classes[1]['probability']:.1f}%" if len(top_classes) > 1 else ""}

### What This Means for You:
Your scan looks healthy! The AI didn't find any suspicious spots, masses, or unusual patterns that would indicate cancer.

### Next Steps:
- Continue with your regular health check-ups
- If you smoke, consider quitting (it's the best thing you can do for your lungs)
- If you have any breathing concerns, talk to your doctor

**Remember:** This is just an AI analysis. Always discuss your results with your healthcare provider who can give you the complete picture."""
    
    else:
        # For any abnormal findings
        return f"""## Your Lung CT Scan Results - Easy Understanding

### What the AI Found:
The analysis detected **{classification}** with **{confidence:.1f}%** confidence. This means the AI found some patterns in your scan that may need further medical attention.

### Understanding the Numbers:
Think of it this way: if we had 100 similar scans, the AI is **{confidence:.1f}%** confident this shows {classification}. Here's the full breakdown:
{chr(10).join([f"- {cls['name']}: {cls['probability']:.1f}%" for cls in top_classes])}

### What This Actually Means:
The AI found areas in your lungs that look different from healthy tissue. These are sometimes called "regions of interest" - think of them like warning flags that caught the AI's attention.

### Important Notes:
- **This is NOT a final diagnosis** - Only a doctor can confirm what these findings mean
- The AI confidence of {confidence:.1f}% means it's quite sure, but medical professionals need to review
- Further tests may be needed to understand exactly what these findings are

### What You Should Do:
1. **Don't panic** - Many abnormal findings turn out to be benign (not cancer)
2. **See a specialist** - A pulmonologist (lung doctor) should review these results
3. **Ask questions** - Write down any questions you have for your doctor

**Most Important:** Schedule a follow-up with your healthcare provider as soon as possible. They can determine if additional tests are needed and what the next steps should be."""


@app.post("/gemini-report", response_model=GeminiReportResponse)
async def generate_gemini_report_endpoint(request: GeminiReportRequest):
    """
    Generate a layman's report using Gemini API.
    """
    try:
        analysis_data = {
            "classification": request.classification,
            "confidence": request.confidence,
            "classes": request.classes,
            "analysis": request.analysis,
            "regions": request.regions
        }
        
        report = await generate_gemini_report(analysis_data)
        
        return GeminiReportResponse(
            success=True,
            report=report,
            timestamp=datetime.utcnow().isoformat()
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to generate Gemini report: {str(e)}"
        )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

