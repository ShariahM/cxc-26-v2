from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import aiofiles
import uuid
import os
from pathlib import Path
from typing import Dict, Optional
import asyncio
from datetime import datetime

from app.services.video_processor import VideoProcessor
from app.services.feedback_generator import FeedbackGenerator

app = FastAPI(
    title="NFL Video Analysis API",
    description="API for analyzing NFL gameplay videos with AI",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173", "http://localhost:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Create necessary directories
UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")
UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# Task storage (in production, use Redis or a database)
tasks: Dict[str, dict] = {}

# Initialize processors
video_processor = VideoProcessor()
feedback_generator = FeedbackGenerator()


@app.get("/")
async def root():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "service": "NFL Video Analysis API",
        "version": "1.0.0"
    }


@app.post("/api/upload")
async def upload_video(file: UploadFile = File(...)):
    """
    Upload a video file for processing
    Only accepts MP4 files
    """
    # Validate file type
    if not file.filename.endswith('.mp4'):
        raise HTTPException(
            status_code=400,
            detail="Only MP4 files are allowed"
        )
    
    # Generate unique task ID
    task_id = str(uuid.uuid4())
    
    # Save uploaded file
    file_path = UPLOAD_DIR / f"{task_id}.mp4"
    chunk_size = 1024 * 1024  # 1MB chunks
    
    try:
        async with aiofiles.open(file_path, 'wb') as f:
            while True:
                chunk = await file.read(chunk_size)
                if not chunk:
                    break
                await f.write(chunk)
        await file.close()
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Failed to save file: {str(e)}"
        )
    
    # Initialize task
    tasks[task_id] = {
        "id": task_id,
        "status": "queued",
        "filename": file.filename,
        "uploaded_at": datetime.now().isoformat(),
        "progress": 0,
        "message": "Video uploaded successfully, processing will begin shortly"
    }
    
    # Start processing in background
    asyncio.create_task(process_video(task_id, file_path))
    
    return {
        "task_id": task_id,
        "status": "queued",
        "message": "Video uploaded successfully and queued for processing"
    }


async def process_video(task_id: str, video_path: Path):
    """Background task to process video"""
    try:
        # Update status
        tasks[task_id]["status"] = "processing"
        tasks[task_id]["message"] = "Processing video..."
        
        # Process video with detection, tracking, and openscore calculation
        results = await video_processor.process(
            str(video_path),
            task_id,
            progress_callback=lambda p: update_progress(task_id, p)
        )
        
        # Generate feedback based on analysis
        tasks[task_id]["message"] = "Generating feedback..."
        feedback = feedback_generator.generate(results)
        
        # Save output video path
        output_path = OUTPUT_DIR / f"{task_id}_annotated.mp4"
        
        # Update task with results
        tasks[task_id].update({
            "status": "completed",
            "progress": 100,
            "message": "Processing completed successfully",
            "results": {
                "total_frames": results["total_frames"],
                "fps": results["fps"],
                "duration": results["duration"],
                "players_detected": results["players_detected"],
                "tracking_data": results["tracking_summary"],
                "openscore_data": results["openscore_summary"],
                "feedback": feedback,
                "output_video": str(output_path)
            },
            "completed_at": datetime.now().isoformat()
        })
        
    except Exception as e:
        tasks[task_id].update({
            "status": "failed",
            "message": f"Processing failed: {str(e)}",
            "error": str(e)
        })


def update_progress(task_id: str, progress: int):
    """Update task progress"""
    if task_id in tasks:
        tasks[task_id]["progress"] = progress


@app.get("/api/status/{task_id}")
async def get_status(task_id: str):
    """Get processing status for a task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    return {
        "task_id": task_id,
        "status": task["status"],
        "progress": task.get("progress", 0),
        "message": task.get("message", ""),
        "uploaded_at": task.get("uploaded_at"),
        "completed_at": task.get("completed_at")
    }


@app.get("/api/results/{task_id}")
async def get_results(task_id: str):
    """Get analysis results for a completed task"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail=f"Task is not completed yet. Current status: {task['status']}"
        )
    
    return {
        "task_id": task_id,
        "results": task["results"]
    }


@app.get("/api/download/{task_id}")
async def download_video(task_id: str):
    """Download the annotated video"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    task = tasks[task_id]
    
    if task["status"] != "completed":
        raise HTTPException(
            status_code=400,
            detail="Video processing not completed yet"
        )
    
    output_path = task["results"]["output_video"]
    
    if not os.path.exists(output_path):
        raise HTTPException(status_code=404, detail="Output video not found")
    
    return FileResponse(
        output_path,
        media_type="video/mp4",
        filename=f"analyzed_{task['filename']}",
        content_disposition_type="inline"
    )


@app.delete("/api/task/{task_id}")
async def delete_task(task_id: str):
    """Delete a task and its associated files"""
    if task_id not in tasks:
        raise HTTPException(status_code=404, detail="Task not found")
    
    # Delete files
    video_path = UPLOAD_DIR / f"{task_id}.mp4"
    output_path = OUTPUT_DIR / f"{task_id}_annotated.mp4"
    
    if video_path.exists():
        video_path.unlink()
    if output_path.exists():
        output_path.unlink()
    
    # Remove task from storage
    del tasks[task_id]
    
    return {"message": "Task deleted successfully"}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
