
# Car Damage Analysis Backend (Django + YOLOv8)

This repository contains the **backend service** for the Car Damage **Detection & Analysis System**.
It is built with **Django REST Framework** and integrates a **YOLOv8 segmentation model** to not only **detect damaged regions** but also:

* ✅ Identify multiple types of damage (scratches, dents, broken glass, etc.)
* ✅ Generate **segmentation masks** and **bounding boxes**
* ✅ Compute **damage percentage** based on mask pixel coverage
* ✅ Provide **confidence scores** for detections
* ✅ Save original + annotated images in the database
* ✅ Return a structured JSON response for frontend visualization

---

## 📂 Project Structure

```
backend_project/
│
├── backend_project/          # Main Django project (settings, configs)
│   ├── __init__.py
│   ├── asgi.py               # Entry point for ASGI servers
│   ├── settings.py           # Django settings (DB, media, installed apps, etc.)
│   ├── urls.py               # Root API routing
│   └── wsgi.py               # Entry point for WSGI servers (Gunicorn/uWSGI)
│
├── detection/                # Core app for damage detection
│   ├── migrations/           # Database migration files
│   ├── __init__.py
│   ├── admin.py              # Django admin panel registration
│   ├── apps.py               # App configuration
│   ├── models.py             # Database model (DetectionResult)
│   ├── urls.py               # Routes for API endpoints
│   ├── views.py              # Business logic for YOLO inference & response
│   └── tests.py              # Unit tests (placeholder)
│
├── media/                    # Uploaded + annotated images (runtime generated)
│
├── manage.py                 # Django CLI entry point
├── requirements.txt          # Python dependencies
└── README.md                 # Project documentation
```

---

## ⚙️ Tech Stack

* **Framework:** Django 5.x, Django REST Framework
* **AI Model:** YOLOv8 (Ultralytics) – segmentation model (`best.pt`)
* **Database:** SQLite (default) or configurable to PostgreSQL/MySQL
* **Image Processing:** OpenCV, NumPy
* **Storage:** Django File Storage (Media folder)

---

## 🧩 Backend Workflow

1. **API Endpoint**

   * Upload endpoint: `POST /api/analyze/`

2. **YOLO Model Handling**

   * Lazy loads `best.pt` only once on startup.
   * Uses segmentation masks (`masks`) to detect exact damaged areas.

3. **Image Processing**

   * Saves uploaded image temporarily
   * Runs YOLO inference
   * Draws:

     * Segmentation masks (colored overlays per damage class)
     * Bounding boxes + labels for each detection

4. **Damage Calculation**

   * Damage % is based on **mask pixel count** vs **total image pixels**
   * Confidence is taken as **highest detection confidence**

5. **Database Storage**

   * Saves both **original image** and **annotated image**
   * Stores detection results in JSON format

6. **Response JSON Example**

```json
{
  "id": 12,
  "overallDamagePercentage": 34.83,
  "overallConfidence": 86.94,
  "detections": [
    {
      "class": "mop_lom",
      "confidence": 86.94,
      "damagePercentage": 34.83,
      "bbox": [
        136.30,
        56.81,
        537.57,
        394.21
      ]
    }
  ],
  "original_image_url": "http://127.0.0.1:8000/media/detections/input.jpg",
  "annotated_image_url": "http://127.0.0.1:8000/media/detections/annotated/output.jpg"
}
```

---

## 🚀 Setup Instructions

### 1. Clone Repository

```bash
git clone https://github.com/your-org/backend.git
cd backend
```

### 2. Create Virtual Environment

```bash
python -m venv venv
source venv/bin/activate   # (Linux/Mac)
venv\Scripts\activate      # (Windows)
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

### 4. Apply Migrations

```bash
python manage.py makemigrations
python manage.py migrate
```

### 5. Run Development Server

```bash
python manage.py runserver
```

Server runs at: **[http://127.0.0.1:8000/](http://127.0.0.1:8000/)**

---

## 🖼️ Example Usage

### API Request:

```bash
curl -X POST http://127.0.0.1:8000/api/analyze/ \
  -F "file=@car.jpg"
```

### API Response:

(See sample JSON above)

---

## 📊 Features Implemented

* ✅ YOLOv8 Segmentation Model integrated
* ✅ Mask + Bounding Box visualization
* ✅ Pixel-based Damage Percentage calculation
* ✅ Confidence score extraction
* ✅ Automatic annotated image generation
* ✅ Database storage of results
* ✅ REST API returning structured JSON
* ✅ Media files served locally

---

## 🔮 Next Steps (Future Work)

* Add authentication for API usage
* Switch from SQLite → PostgreSQL for scalability
* Dockerize the backend for deployment
* Integrate Celery for async model inference
* Optimize YOLO model (quantization / pruning)

---

## 👨‍💻 Developer Notes

* **WSGI vs ASGI**:

  * `wsgi.py` → Used for production (Gunicorn/uWSGI)
  * `asgi.py` → For async servers (Daphne, Uvicorn, etc.)

* **Damage Calculation**:

  * Strictly **mask-based** (not bounding box-based)
  * Ensures accurate measurement of damaged area

---

🔑 This backend is now **fully functional** and ready for integration with the frontend.

