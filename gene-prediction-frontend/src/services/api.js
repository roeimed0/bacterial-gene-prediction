// API service for bacterial gene prediction backend

const API_BASE_URL = 'http://localhost:8000';

export const api = {
  // Health check
  async checkHealth() {
    const response = await fetch(`${API_BASE_URL}/health`);
    if (!response.ok) throw new Error('API health check failed');
    return response.json();
  },

  // Predict genes from sequence
  async predictGenes(sequence, options = {}) {
    const {
      useGroupML = true,
      groupThreshold = 0.1,
      useFinalML = true,
      finalThreshold = 0.12
    } = options;

    const response = await fetch(`${API_BASE_URL}/predict`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        sequence,
        use_group_ml: useGroupML,
        group_threshold: groupThreshold,
        use_final_ml: useFinalML,
        final_threshold: finalThreshold
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Prediction failed');
    }

    return response.json();
  },

  // Predict genes from uploaded file
  async predictFromFile(file, options = {}) {
    const {
      useGroupML = true,
      groupThreshold = 0.1,
      useFinalML = true,
      finalThreshold = 0.12
    } = options;

    const formData = new FormData();
    formData.append('file', file);
    formData.append('use_group_ml', useGroupML);
    formData.append('group_threshold', groupThreshold);
    formData.append('use_final_ml', useFinalML);
    formData.append('final_threshold', finalThreshold);

    const response = await fetch(`${API_BASE_URL}/predict/file`, {
      method: 'POST',
      body: formData
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Prediction failed');
    }

    return response.json();
  }
};