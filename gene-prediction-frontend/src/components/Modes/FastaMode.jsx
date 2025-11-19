import { useState } from 'react';
import { Upload, FileText, Loader } from 'lucide-react';
import { api } from '../../services/api';
import ResultsView from '../Results/ResultsView';

const FastaMode = ({ results, setResults, addJob, removeJob }) => {
  const [file, setFile] = useState(null);
  const [pastedSequence, setPastedSequence] = useState('');
  const [uploadMethod, setUploadMethod] = useState('file'); // 'file' or 'paste'
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentJobId, setCurrentJobId] = useState(null);

  // ML Options
  const [useGroupML, setUseGroupML] = useState(true);
  const [useFinalML, setUseFinalML] = useState(true);
  const [groupThreshold, setGroupThreshold] = useState(0.1);
  const [finalThreshold, setFinalThreshold] = useState(0.12);

  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResults(null);

    const options = {
      useGroupML,
      groupThreshold,
      useFinalML,
      finalThreshold
    };

    setLoading(true);

    if (uploadMethod === 'file') {
      if (!file) {
        setError('Please select a FASTA file');
        setLoading(false);
        return;
      }

      const jobId = addJob({
        name: file.name,
        mode: 'FASTA Upload',
        type: 'prediction'
      });
      setCurrentJobId(jobId);

      try {
        const response = await api.predictFromFile(file, options);
        setResults(response);
        removeJob(jobId);
        setCurrentJobId(null);
      } catch (err) {
        setError(err.message || 'Prediction failed');
        removeJob(jobId);
        setCurrentJobId(null);
      } finally {
        setLoading(false);
      }
    } else {
      if (!pastedSequence.trim()) {
        setError('Please paste a DNA sequence');
        setLoading(false);
        return;
      }

      const jobId = addJob({
        name: 'Pasted Sequence',
        mode: 'FASTA Paste',
        type: 'prediction'
      });
      setCurrentJobId(jobId);

      try {
        const response = await api.predictGenes(pastedSequence, options);
        setResults(response);
        removeJob(jobId);
        setCurrentJobId(null);
      } catch (err) {
        setError(err.message || 'Prediction failed');
        removeJob(jobId);
        setCurrentJobId(null);
      } finally {
        setLoading(false);
      }
    }
  };

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Upload Method Toggle */}
        <div className="flex gap-2">
          <button
            type="button"
            onClick={() => setUploadMethod('file')}
            className={`flex-1 py-3 px-4 rounded-lg font-medium transition-colors ${
              uploadMethod === 'file'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <Upload className="w-5 h-5 inline mr-2" />
            Upload File
          </button>
          <button
            type="button"
            onClick={() => setUploadMethod('paste')}
            className={`flex-1 py-3 px-4 rounded-lg font-medium transition-colors ${
              uploadMethod === 'paste'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <FileText className="w-5 h-5 inline mr-2" />
            Paste Sequence
          </button>
        </div>

        {/* File Upload */}
        {uploadMethod === 'file' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Select FASTA File:
            </label>
            <input
              type="file"
              accept=".fasta,.fa,.fna"
              onChange={handleFileChange}
              disabled={loading}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
            />
            {file && (
              <p className="text-sm text-gray-600 mt-2">
                Selected: {file.name} ({(file.size / 1024).toFixed(1)} KB)
              </p>
            )}
          </div>
        )}

        {/* Paste Sequence */}
        {uploadMethod === 'paste' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Paste DNA Sequence:
            </label>
            <textarea
              value={pastedSequence}
              onChange={(e) => setPastedSequence(e.target.value)}
              placeholder=">sequence_name&#10;ATGCGATCGATCG..."
              disabled={loading}
              rows={10}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100 font-mono text-sm"
            />
            <p className="text-xs text-gray-500 mt-1">
              You can paste raw sequence or FASTA format (with header)
            </p>
          </div>
        )}

        {/* ML Options */}
        <div className="bg-gray-50 rounded-lg p-6 space-y-4">
          <h3 className="font-semibold text-gray-900">
            Machine Learning Options
          </h3>

          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={useGroupML}
              onChange={(e) => setUseGroupML(e.target.checked)}
              disabled={loading}
              className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700">
              Use ML Group Filtering
            </span>
          </label>

          {useGroupML && (
            <div className="ml-7 flex items-center gap-3">
              <label className="text-sm text-gray-600">Threshold:</label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={groupThreshold}
                onChange={(e) => setGroupThreshold(parseFloat(e.target.value))}
                disabled={loading}
                className="w-24 px-3 py-1 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
              />
            </div>
          )}

          <label className="flex items-center gap-3 cursor-pointer">
            <input
              type="checkbox"
              checked={useFinalML}
              onChange={(e) => setUseFinalML(e.target.checked)}
              disabled={loading}
              className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700">
              Use Final ML Filtration
            </span>
          </label>

          {useFinalML && (
            <div className="ml-7 flex items-center gap-3">
              <label className="text-sm text-gray-600">Threshold:</label>
              <input
                type="number"
                step="0.01"
                min="0"
                max="1"
                value={finalThreshold}
                onChange={(e) => setFinalThreshold(parseFloat(e.target.value))}
                disabled={loading}
                className="w-24 px-3 py-1 border border-gray-300 rounded focus:ring-2 focus:ring-blue-500"
              />
            </div>
          )}
        </div>

        {/* Submit Button */}
        <button
          type="submit"
          disabled={loading}
          className="w-full py-3 px-6 bg-blue-600 text-white font-medium rounded-lg hover:bg-blue-700 transition-colors disabled:bg-gray-400 disabled:cursor-not-allowed flex items-center justify-center gap-2"
        >
          {loading ? (
            <>
              <Loader className="w-5 h-5 animate-spin" />
              Predicting Genes...
            </>
          ) : (
            <>
              <Upload className="w-5 h-5" />
              Predict Genes
            </>
          )}
        </button>
      </form>

      {/* Error Display */}
      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800 font-medium">❌ {error}</p>
        </div>
      )}

      {/* Results Display */}
      {results && <ResultsView results={results} mode="fasta" />}
    </div>
  );
};

export default FastaMode;