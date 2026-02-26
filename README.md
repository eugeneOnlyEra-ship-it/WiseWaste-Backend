# WasteWise

> Photograph your waste. Know exactly how to dispose of it.


---

## Project Structure

```
wastewise/
├── backend/
│   ├── main.py                     (FastAPI server)
│   ├── requirements.txt            (Python dependencies)
│   └── wastewise_model/            (Place your trained model here)
│       ├── model_weights.pth
│       ├── model_traced.pt
│       └── config.json
└── frontend/
    ├── src/
    │   ├── App.jsx           (Main React component)
    │   └── main.jsx          (Entry point)
    ├── index.html
    ├── package.json
    └── vite.config.js
```

---

## Setup Instructions

### Step 1 — Place your trained model

Copy the `wastewise_model/` folder you downloaded from Colab into the `backend/` directory:

```bash
cp -r ~/Downloads/wastewise_model/ backend/
```

Your backend folder should now look like:
```
backend/
├── main.py
├── requirements.txt
└── wastewise_model/
    ├── model_weights.pth
    ├── model_traced.pt
    └── config.json
```

---

### Step 2 — Start the Backend

Open a terminal and run:

```bash
cd backend
pip install -r requirements.txt
uvicorn main:app --reload --port 8000
```

You should see:
```
Model loaded | Classes: ['cardboard', 'glass', 'metal', 'paper', 'plastic', 'trash']
INFO:     Uvicorn running on http://127.0.0.1:8000
```

Test it's working: open http://localhost:8000 in your browser. You should see:
```json
{"message": "WasteWise API is running ", "version": "1.0.0"}
```

---

### Step 3 — Start the Frontend

Open a **second terminal** and run:

```bash
cd frontend
npm install
npm run dev
```

You should see:
```
  VITE v5.x  ready in xxx ms
  ➜  Local:   http://localhost:5173/
```

Open http://localhost:5173 in your browser — WasteWise is running!

---

## How to Use

1. Open http://localhost:5173
2. Drag and drop a photo of waste, or click to upload
3. Click **Classify Waste**
4. See the category, confidence score, and disposal instructions

---

## Tech Stack

- **ML Model:** MobileNetV2 fine-tuned on TrashNet (~85-90% accuracy)
- **Backend:** FastAPI + PyTorch
- **Frontend:** React + Vite
- **Dataset:** TrashNet (2,500+ images, 6 categories)

---

## Categories

| Category | Disposal |
|----------|----------|
|  Cardboard | Flatten & Recycle |
| Glass | Glass Recycling Bin |
| Metal | Rinse & Recycle |
|  Paper | Paper Recycling Bin |
|  Plastic | Check Symbol & Recycle |
| Trash | General Waste |
# WiseWaste-Backend
