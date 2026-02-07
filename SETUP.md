# Setup Guide - NFL Video Analysis Webapp

This guide will help you set up and run the NFL Video Analysis application.

## Prerequisites

- **Python 3.8+** installed
- **Node.js 16+** and npm installed
- Your trained YOLOv8 model file (`.pt` format)

## Backend Setup

### 1. Navigate to the backend directory

```bash
cd backend
```

### 2. Create a virtual environment (recommended)

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# macOS/Linux
python3 -m venv venv
source venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Place your YOLOv8 model

Create a `models` directory and place your trained YOLOv8 model there:

```bash
mkdir models
# Copy your model file (e.g., best.pt) to backend/models/
```

**Important:** Update the model path in `backend/app/models/detection.py` if your model has a different name:

```python
# Line 11 in detection.py
def __init__(self, model_path: str = "models/your_model_name.pt"):
```

### 5. Run the backend server

```bash
# From the backend directory
uvicorn app.main:app --reload
```

The backend API will be available at `http://localhost:8000`

You can test the API by visiting `http://localhost:8000/docs` (FastAPI automatic documentation)

## Frontend Setup

### 1. Open a new terminal and navigate to the frontend directory

```bash
cd frontend
```

### 2. Install dependencies

```bash
npm install
```

### 3. Run the development server

```bash
npm run dev
```

The frontend will be available at `http://localhost:5173`

## Usage

1. **Start the backend** (in one terminal):
   ```bash
   cd backend
   uvicorn app.main:app --reload
   ```

2. **Start the frontend** (in another terminal):
   ```bash
   cd frontend
   npm run dev
   ```

3. **Access the app** at `http://localhost:5173`

4. **Upload a video**:
   - Drag and drop an MP4 file or click to browse
   - Wait for processing (processing time depends on video length)
   - View analysis results, feedback, and download annotated video

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── models/
│   │   │   ├── detection.py     # YOLOv8 detection
│   │   │   ├── tracking.py      # ByteTrack player tracking
│   │   │   └── openscore.py     # OpenScore calculation
│   │   └── services/
│   │       ├── video_processor.py
│   │       └── feedback_generator.py
│   ├── requirements.txt
│   ├── uploads/                 # Created automatically
│   ├── outputs/                 # Created automatically
│   └── models/                  # Place your YOLOv8 model here
│
├── frontend/
│   ├── src/
│   │   ├── App.jsx              # Main application component
│   │   ├── components/
│   │   │   ├── VideoUpload.jsx  # Drag-and-drop upload
│   │   │   └── AnalysisResults.jsx
│   │   └── services/
│   │       └── api.js           # API client
│   └── package.json
│
└── README.md
```

## API Endpoints

- **POST /api/upload** - Upload video file
- **GET /api/status/{task_id}** - Check processing status
- **GET /api/results/{task_id}** - Get analysis results
- **GET /api/download/{task_id}** - Download annotated video
- **DELETE /api/task/{task_id}** - Delete task and files

## Customization

### Adjusting YOLOv8 Class Names

If your model has different class names, update them in `backend/app/models/detection.py`:

```python
self.class_names = {
    0: 'your_class_0',
    1: 'your_class_1',
    # ... add your classes
}
```

### Adjusting OpenScore Weights

You can modify the scoring weights in `backend/app/models/openscore.py`:

```python
self.weights = {
    'distance': 0.4,      # Distance from defenders
    'velocity': 0.25,     # Defender approach velocity
    'separation': 0.25,   # Route separation
    'coverage': 0.1       # Coverage scheme
}
```

### Changing Video Processing Settings

Modify detection confidence threshold in `backend/app/services/video_processor.py`:

```python
detections = self.detector.detect(frame, conf_threshold=0.3)  # Adjust threshold
```

## Troubleshooting

### Backend Issues

1. **Module not found errors**: Make sure virtual environment is activated and all dependencies are installed
2. **Model not loading**: Verify model path and ensure the file exists
3. **Port already in use**: Change port in uvicorn command: `uvicorn app.main:app --reload --port 8001`

### Frontend Issues

1. **CORS errors**: Check that backend CORS settings in `main.py` match your frontend URL
2. **npm install fails**: Try deleting `node_modules` and `package-lock.json`, then run `npm install` again
3. **Port 5173 in use**: Vite will automatically use the next available port

### Processing Issues

1. **Video processing fails**: Check video format (must be MP4) and ensure CUDA/GPU support is configured if needed
2. **Slow processing**: Consider reducing video resolution or using GPU acceleration
3. **Out of memory**: Process shorter video clips or reduce batch size in detection

## Performance Optimization

- **GPU Acceleration**: Install CUDA-enabled PyTorch for faster processing
- **Batch Processing**: Adjust batch size in video processor for your hardware
- **Video Resolution**: Resize videos to lower resolution for faster processing

## Production Deployment

For production deployment:

1. **Backend**: Use a production WSGI server like Gunicorn
   ```bash
   gunicorn app.main:app -w 4 -k uvicorn.workers.UvicornWorker
   ```

2. **Frontend**: Build the production bundle
   ```bash
   npm run build
   ```
   Serve the `dist` folder with a web server like Nginx

3. **Database**: Replace in-memory task storage with Redis or a database

4. **File Storage**: Use cloud storage (S3, Azure Storage) for video files

## License

MIT
