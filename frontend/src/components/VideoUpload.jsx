import React, { useCallback, useState } from 'react';
import { useDropzone } from 'react-dropzone';

const VideoUpload = ({ onUploadSuccess, onUploadError }) => {
  const [isUploading, setIsUploading] = useState(false);
  const [uploadError, setUploadError] = useState(null);

  const onDrop = useCallback(async (acceptedFiles, rejectedFiles) => {
    // Clear previous errors
    setUploadError(null);

    // Check for rejected files
    if (rejectedFiles.length > 0) {
      const error = rejectedFiles[0].errors[0];
      if (error.code === 'file-invalid-type') {
        setUploadError('Only MP4 files are allowed');
      } else if (error.code === 'too-many-files') {
        setUploadError('Please upload only one file at a time');
      } else {
        setUploadError(error.message);
      }
      if (onUploadError) {
        onUploadError(error.message);
      }
      return;
    }

    // Check if we have a valid file
    if (acceptedFiles.length === 0) {
      return;
    }

    const file = acceptedFiles[0];
    setIsUploading(true);

    try {
      // Import the API function
      const { uploadVideo } = await import('../services/api.js');
      
      // Upload the video
      const response = await uploadVideo(file);
      
      if (onUploadSuccess) {
        onUploadSuccess(response);
      }
    } catch (error) {
      const errorMessage = error.response?.data?.detail || error.message || 'Upload failed';
      setUploadError(errorMessage);
      if (onUploadError) {
        onUploadError(errorMessage);
      }
    } finally {
      setIsUploading(false);
    }
  }, [onUploadSuccess, onUploadError]);

  const { getRootProps, getInputProps, isDragActive } = useDropzone({
    onDrop,
    accept: {
      'video/mp4': ['.mp4']
    },
    maxFiles: 1,
    multiple: false,
    disabled: isUploading
  });

  return (
    <div className="video-upload-container">
      <div
        {...getRootProps()}
        className={`dropzone ${isDragActive ? 'active' : ''} ${isUploading ? 'uploading' : ''}`}
      >
        <input {...getInputProps()} />
        
        <div className="dropzone-content">
          {isUploading ? (
            <>
              <div className="spinner"></div>
              <p>Uploading video...</p>
            </>
          ) : isDragActive ? (
            <>
              <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p className="primary-text">Drop the MP4 file here</p>
            </>
          ) : (
            <>
              <svg className="upload-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
              </svg>
              <p className="primary-text">Drag & drop your NFL game video here</p>
              <p className="secondary-text">or click to browse</p>
              <p className="format-text">Only MP4 files are supported</p>
            </>
          )}
        </div>
      </div>

      {uploadError && (
        <div className="error-message">
          <svg className="error-icon" viewBox="0 0 24 24" fill="currentColor">
            <path d="M12 2C6.48 2 2 6.48 2 12s4.48 10 10 10 10-4.48 10-10S17.52 2 12 2zm1 15h-2v-2h2v2zm0-4h-2V7h2v6z" />
          </svg>
          {uploadError}
        </div>
      )}
    </div>
  );
};

export default VideoUpload;
