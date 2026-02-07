import axios from 'axios';

const API_BASE_URL = 'http://localhost:8000';

// Create axios instance with default config
const apiClient = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

/**
 * Upload video file for analysis
 * @param {File} file - MP4 video file
 * @returns {Promise} - Upload response with task_id
 */
export const uploadVideo = async (file) => {
  const formData = new FormData();
  formData.append('file', file);

  const response = await apiClient.post('/api/upload', formData, {
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  return response.data;
};

/**
 * Check processing status
 * @param {string} taskId - Task ID from upload
 * @returns {Promise} - Status information
 */
export const getStatus = async (taskId) => {
  const response = await apiClient.get(`/api/status/${taskId}`);
  return response.data;
};

/**
 * Get analysis results
 * @param {string} taskId - Task ID from upload
 * @returns {Promise} - Analysis results
 */
export const getResults = async (taskId) => {
  const response = await apiClient.get(`/api/results/${taskId}`);
  return response.data;
};

/**
 * Get download URL for annotated video
 * @param {string} taskId - Task ID from upload
 * @returns {string} - Download URL
 */
export const getDownloadUrl = (taskId) => {
  return `${API_BASE_URL}/api/download/${taskId}`;
};

/**
 * Delete task and associated files
 * @param {string} taskId - Task ID to delete
 * @returns {Promise} - Deletion confirmation
 */
export const deleteTask = async (taskId) => {
  const response = await apiClient.delete(`/api/task/${taskId}`);
  return response.data;
};

export default {
  uploadVideo,
  getStatus,
  getResults,
  getDownloadUrl,
  deleteTask,
};
