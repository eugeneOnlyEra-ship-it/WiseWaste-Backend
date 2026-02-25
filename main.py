from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import torch
import torchvision.transforms as transforms
from torchvision import models
import torch.nn as nn
from PIL import Image
import io
import json
import os

app = FastAPI(title="WasteWise API", version="1.0.0")

#after importing libraries
# Allow frontend to talk to backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Disposal instructions per category ────────────────────────────────────
DISPOSAL_INFO = {
    "cardboard": {
        "bin": "Recycling Bin ♻️",
        "color": "#3B82F6",
        "tip": "Flatten boxes before placing in the recycling bin. Remove any tape or staples if possible.",
        "emoji": "📦"
    },
    "glass": {
        "bin": "Glass Recycling 🫙",
        "color": "#10B981",
        "tip": "Rinse the container and place it in the glass recycling bin. Do not mix with broken glass — wrap that in newspaper and place in general waste.",
        "emoji": "🫙"
    },
    "metal": {
        "bin": "Recycling Bin ♻️",
        "color": "#F59E0B",
        "tip": "Rinse cans and tins before recycling. Aluminium foil can also be recycled if clean.",
        "emoji": "🥫"
    },
    "paper": {
        "bin": "Recycling Bin ♻️",
        "color": "#8B5CF6",
        "tip": "Place clean, dry paper in the recycling bin. Greasy or food-stained paper (like pizza boxes) should go in general waste.",
        "emoji": "📄"
    },
    "plastic": {
        "bin": "Recycling Bin ♻️",
        "color": "#EC4899",
        "tip": "Check the recycling symbol on the bottom. Rinse containers before recycling. Plastic bags usually cannot be recycled kerbside.",
        "emoji": "🧴"
    },
    "trash": {
        "bin": "General Waste 🗑️",
        "color": "#6B7280",
        "tip": "This item cannot be recycled. Place it in your general waste bin. Consider if a reusable alternative exists for next time.",
        "emoji": "🗑️"
    }
}

# ── Routes ─────────────────────────────────────────────────────────────────
@app.get("/")
def root():
    return {"message": "WasteWise API is running 🌱", "version": "1.0.0"}

@app.get("/health")
def health():
    return {"status": "ok", "classes": CLASS_NAMES}

@app.post("/classify")
async def classify(file: UploadFile = File(...)):
    # Validate file type
    if file.content_type not in ["image/jpeg", "image/png", "image/webp", "image/jpg"]:
        raise HTTPException(status_code=400, detail="Only JPEG, PNG, and WebP images are supported.")

    # Read and preprocess image
    try:
        contents = await file.read()
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not read image. Please try another file.")

    # Run inference
    tensor = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    top_idx = probs.argmax().item()
    predicted_class = CLASS_NAMES[top_idx]
    confidence = round(probs[top_idx].item() * 100, 1)

    # Build all class probabilities
    all_probs = {
        CLASS_NAMES[i]: round(probs[i].item() * 100, 1)
        for i in range(len(CLASS_NAMES))
    }

    disposal = DISPOSAL_INFO.get(predicted_class, {
        "bin": "General Waste",
        "color": "#6B7280",
        "tip": "When in doubt, place in general waste.",
        "emoji": "🗑️"
    })

    return JSONResponse({
        "predicted_class": predicted_class,
        "confidence": confidence,
        "all_probabilities": all_probs,
        "disposal": disposal
    })
