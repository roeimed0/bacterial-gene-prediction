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
  },

  // Predict genes from NCBI accession
  async predictFromNcbi(accession, email, options = {}) {
    const {
      useGroupML = true,
      groupThreshold = 0.1,
      useFinalML = true,
      finalThreshold = 0.12
    } = options;

    const response = await fetch(`${API_BASE_URL}/predict/ncbi`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        accession,
        email,
        use_group_ml: useGroupML,
        group_threshold: groupThreshold,
        use_final_ml: useFinalML,
        final_threshold: finalThreshold
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'NCBI download failed');
    }

    return response.json();
  },

  // Get genome catalog
  async getCatalog() {
    const response = await fetch(`${API_BASE_URL}/catalog`);
    if (!response.ok) throw new Error('Failed to load catalog');
    return response.json();
  },

  // Get available results for validation
  async getResults() {
    const response = await fetch(`${API_BASE_URL}/results`);
    if (!response.ok) throw new Error('Failed to load results');
    return response.json();
  },

  // Validate predictions
  async validatePredictions(genomeId) {
    const response = await fetch(`${API_BASE_URL}/validate`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        genome_id: genomeId
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Validation failed');
    }

    return response.json();
  },
   async getFiles() {
    const response = await fetch(`${API_BASE_URL}/files`);
    if (!response.ok) throw new Error('Failed to load files');
    return response.json();
  },

  // Delete specific files
  async deleteFiles(paths) {
    const response = await fetch(`${API_BASE_URL}/files/delete`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        paths
      })
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Delete failed');
    }

    return response.json();
  },

  // Delete all files
  async cleanupAllFiles() {
    const response = await fetch(`${API_BASE_URL}/files/cleanup`, {
      method: 'POST',
    });

    if (!response.ok) {
      const error = await response.json();
      throw new Error(error.detail || 'Cleanup failed');
    }

    return response.json();
  }
};