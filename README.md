# NFL Video Analysis Webapp

A comprehensive video analysis tool for NFL quarterbacks that uses AI to track players, calculate open scores, and provide feedback on passing decisions.

## Features

- **Video Upload**: Drag and drop MP4 file upload
- **Player Detection**: YOLOv8-based player detection
- **Player Tracking**: Automatic player tracking across frames
- **Open Score Calculation**: Real-time calculation of receiver openness
- **QB Feedback**: Intelligent feedback on passing decisions

## Tech Stack

- **Frontend**: React with drag-and-drop file upload
- **Backend**: FastAPI for video processing
- **Detection**: YOLOv8 (custom model)
- **Tracking**: ByteTrack algorithm
- **Processing**: OpenCV, NumPy

## Project Structure

```
.
├── backend/
│   ├── app/
│   │   ├── main.py              # FastAPI application
│   │   ├── models/
│   │   │   ├── detection.py     # YOLOv8 detection
│   │   │   ├── tracking.py      # Player tracking
│   │   │   └── openscore.py     # OpenScore calculation
│   │   ├── services/
│   │   │   ├── video_processor.py
│   │   │   └── feedback_generator.py
│   │   └── utils/
│   │       └── helpers.py
│   ├── requirements.txt
│   └── models/                  # Place your YOLOv8 model here
├── frontend/
│   ├── src/
│   │   ├── App.jsx
│   │   ├── components/
│   │   │   ├── VideoUpload.jsx
│   │   │   └── AnalysisResults.jsx
│   │   └── services/
│   │       └── api.js
│   └── package.json
└── README.md
```

## Setup

### Backend Setup

```bash
cd backend
pip install -r requirements.txt
# Place your YOLOv8 model in backend/models/
uvicorn app.main:app --reload
```

### Frontend Setup

```bash
cd frontend
npm install
npm run dev
```

## Usage

1. Start the backend server (runs on http://localhost:8000)
2. Start the frontend development server (runs on http://localhost:5173)
3. Drag and drop an MP4 file into the upload area
4. Wait for processing to complete
5. Review the analysis results and feedback

## API Endpoints

- `POST /api/upload` - Upload video file
- `GET /api/status/{task_id}` - Check processing status
- `GET /api/results/{task_id}` - Get analysis results
- `GET /api/download/{task_id}` - Download annotated video

## OpenScore Algorithm

The OpenScore is calculated based on:
- Distance from nearest defender
- Defender velocity and direction
- Route separation
- Coverage scheme detection
- Time to throw consideration

## License

MIT
