import { useState, useEffect } from 'react';
import { CheckCircle, Loader, RefreshCw, AlertCircle } from 'lucide-react';
import { api } from '../../services/api';

const ValidateMode = ({ results, setResults, addJob, removeJob }) => {
  const [availableResults, setAvailableResults] = useState([]);
  const [loading, setLoading] = useState(true);
  const [validating, setValidating] = useState(false);
  const [error, setError] = useState(null);
  const [selectedGenome, setSelectedGenome] = useState(null);

  // Load available results on mount
  useEffect(() => {
    loadResults();
  }, []);

  const loadResults = async () => {
    try {
      setLoading(true);
      const data = await api.getResults();
      // Filter to only show NCBI genomes (can be validated)
      const validatable = data.results.filter(r => r.can_validate);
      setAvailableResults(validatable);
    } catch (err) {
      setError(err.message || 'Failed to load results');
    } finally {
      setLoading(false);
    }
  };

  const handleValidate = async (result) => {
    setSelectedGenome(result.genome_id);
    setResults(null);
    setValidating(true);
    setError(null);

    // Add job to tracker
    const jobId = addJob({
      name: `Validating ${result.genome_id}`,
      mode: 'Validation',
      type: 'validation'
    });

    try {
      const validationResults = await api.validatePredictions(result.genome_id);
      setResults(validationResults);
      removeJob(jobId);
    } catch (err) {
      setError(err.message || 'Validation failed');
      removeJob(jobId);
    } finally {
      setValidating(false);
    }
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader className="w-8 h-8 animate-spin text-blue-600" />
        <span className="ml-3 text-gray-600">Loading results...</span>
      </div>
    );
  }

  if (error && !results) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-800 font-medium">❌ {error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <div className="flex gap-3">
          <AlertCircle className="w-5 h-5 text-blue-600 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-blue-900 font-medium">
              Validation compares your predictions to reference annotations
            </p>
            <p className="text-xs text-blue-700 mt-1">
              Only NCBI genomes with available references can be validated
            </p>
          </div>
        </div>
      </div>

      {/* Refresh Button */}
      <div className="flex justify-between items-center">
        <h3 className="text-lg font-semibold text-gray-900">
          Available Results ({availableResults.length})
        </h3>
        <button
          onClick={loadResults}
          disabled={loading}
          className="flex items-center gap-2 px-3 py-2 text-sm text-blue-600 hover:bg-blue-50 rounded-lg transition-colors"
        >
          <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
          Refresh
        </button>
      </div>

      {/* Results List */}
      {availableResults.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
          <p className="text-gray-600">No prediction results found</p>
          <p className="text-sm text-gray-500 mt-2">
            Run predictions first using Catalog, NCBI, or FASTA modes
          </p>
        </div>
      ) : (
        <div className="grid gap-3">
          {availableResults.map((result) => {
            const date = new Date(result.created * 1000);
            const sizeMB = (result.size / 1024).toFixed(1);

            return (
              <div
                key={result.genome_id}
                className="bg-white border border-gray-200 rounded-lg p-4 hover:border-blue-300 hover:shadow-sm transition-all"
              >
                <div className="flex items-center justify-between">
                  <div className="flex-1">
                    <h4 className="font-semibold text-gray-900 mb-1">
                      {result.genome_id}
                    </h4>
                    <div className="flex gap-4 text-xs text-gray-500">
                      <span>{result.filename}</span>
                      <span>{sizeMB} KB</span>
                      <span>{date.toLocaleDateString()} {date.toLocaleTimeString()}</span>
                    </div>
                  </div>
                  <button
                    onClick={() => handleValidate(result)}
                    disabled={validating}
                    className="ml-4 flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors disabled:bg-gray-400"
                  >
                    <CheckCircle className="w-4 h-4" />
                    Validate
                  </button>
                </div>
              </div>
            );
          })}
        </div>
      )}

      {/* Validating Status */}
      {validating && selectedGenome && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center gap-3">
            <Loader className="w-5 h-5 animate-spin text-blue-600" />
            <div>
              <p className="font-medium text-gray-900">
                Validating predictions...
              </p>
              <p className="text-sm text-gray-600">
                {selectedGenome}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && results && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800 font-medium">❌ {error}</p>
        </div>
      )}

      {/* Validation Results */}
      {results && (
        <div className="space-y-6">
          {/* Summary */}
          <div className="bg-green-50 border border-green-200 rounded-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">
              ✅ Validation Complete
            </h2>
            <div className="grid grid-cols-2 gap-4">
              <div>
                <p className="text-sm text-gray-600">Genome ID</p>
                <p className="text-lg font-semibold text-gray-900">
                  {results.genome_id}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Reference File</p>
                <p className="text-sm font-mono text-gray-700 truncate">
                  {results.reference_file.split('/').pop()}
                </p>
              </div>
            </div>
          </div>

          {/* Metrics */}
          <div className="grid grid-cols-3 gap-4">
            {/* Gene Counts */}
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Gene Counts</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Reference:</span>
                  <span className="font-semibold text-gray-900">
                    {results.reference_count.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Predicted:</span>
                  <span className="font-semibold text-gray-900">
                    {results.predicted_count.toLocaleString()}
                  </span>
                </div>
              </div>
            </div>

            {/* Matches */}
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Matches</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-green-600">True Positives:</span>
                  <span className="font-semibold text-green-700">
                    {results.true_positives.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-red-600">False Positives:</span>
                  <span className="font-semibold text-red-700">
                    {results.false_positives.toLocaleString()}
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-orange-600">False Negatives:</span>
                  <span className="font-semibold text-orange-700">
                    {results.false_negatives.toLocaleString()}
                  </span>
                </div>
              </div>
            </div>

            {/* Performance */}
            <div className="bg-white border border-gray-200 rounded-lg p-4">
              <h3 className="text-sm font-semibold text-gray-700 mb-3">Performance</h3>
              <div className="space-y-2">
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Sensitivity:</span>
                  <span className="font-semibold text-blue-700">
                    {(results.sensitivity * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">Precision:</span>
                  <span className="font-semibold text-blue-700">
                    {(results.precision * 100).toFixed(2)}%
                  </span>
                </div>
                <div className="flex justify-between">
                  <span className="text-sm text-gray-600">F1 Score:</span>
                  <span className="font-semibold text-purple-700">
                    {results.f1_score.toFixed(4)}
                  </span>
                </div>
              </div>
            </div>
          </div>

          {/* Visual Performance Bar */}
          <div className="bg-white border border-gray-200 rounded-lg p-6">
            <h3 className="text-sm font-semibold text-gray-700 mb-4">Performance Overview</h3>
            <div className="space-y-4">
              {/* Sensitivity Bar */}
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-600">Sensitivity (Recall)</span>
                  <span className="font-semibold">{(results.sensitivity * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-blue-600 h-3 rounded-full transition-all"
                    style={{ width: `${results.sensitivity * 100}%` }}
                  />
                </div>
              </div>

              {/* Precision Bar */}
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-600">Precision</span>
                  <span className="font-semibold">{(results.precision * 100).toFixed(1)}%</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-green-600 h-3 rounded-full transition-all"
                    style={{ width: `${results.precision * 100}%` }}
                  />
                </div>
              </div>

              {/* F1 Score Bar */}
              <div>
                <div className="flex justify-between text-sm mb-1">
                  <span className="text-gray-600">F1 Score</span>
                  <span className="font-semibold">{results.f1_score.toFixed(4)}</span>
                </div>
                <div className="w-full bg-gray-200 rounded-full h-3">
                  <div
                    className="bg-purple-600 h-3 rounded-full transition-all"
                    style={{ width: `${results.f1_score * 100}%` }}
                  />
                </div>
              </div>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default ValidateMode;