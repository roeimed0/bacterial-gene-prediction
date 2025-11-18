import { useState } from 'react';
import { Upload, FileText, Loader } from 'lucide-react';
import { api } from '../../services/api';

const FastaMode = ({ results, setResults, addJob, removeJob }) => {
  const [inputMethod, setInputMethod] = useState('paste');
  const [sequence, setSequence] = useState('');
  const [file, setFile] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentJobId, setCurrentJobId] = useState(null);

  // ML Options
  const [useGroupML, setUseGroupML] = useState(true);
  const [useFinalML, setUseFinalML] = useState(true);
  const [groupThreshold, setGroupThreshold] = useState(0.1);
  const [finalThreshold, setFinalThreshold] = useState(0.12);

  // Keep your existing handler functions...
  const handleFileChange = (e) => {
    const selectedFile = e.target.files[0];
    if (selectedFile) {
      if (!selectedFile.name.match(/\.(fasta|fa|fna)$/i)) {
        setError('Invalid file type. Please upload a .fasta, .fa, or .fna file.');
        return;
      }
      setFile(selectedFile);
      setError(null);
    }
  };

  const handleDragOver = (e) => {
    e.preventDefault();
  };

  const handleDrop = (e) => {
    e.preventDefault();
    const droppedFile = e.dataTransfer.files[0];
    if (droppedFile) {
      if (!droppedFile.name.match(/\.(fasta|fa|fna)$/i)) {
        setError('Invalid file type. Please upload a .fasta, .fa, or .fna file.');
        return;
      }
      setFile(droppedFile);
      setError(null);
    }
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResults(null);
    setLoading(true);

    const options = {
      useGroupML,
      groupThreshold,
      useFinalML,
      finalThreshold
    };

    // Determine job name
    const jobName = inputMethod === 'upload' && file 
      ? file.name 
      : 'Pasted sequence';

    // Add job to tracker
    const jobId = addJob({
      name: jobName,
      mode: 'FASTA Upload',
      type: 'prediction'
    });
    setCurrentJobId(jobId);

    try {
      let response;

      if (inputMethod === 'paste') {
        if (!sequence.trim()) {
          throw new Error('Please enter a sequence');
        }
        response = await api.predictGenes(sequence, options);
      } else {
        if (!file) {
          throw new Error('Please select a file');
        }
        response = await api.predictFromFile(file, options);
      }

      setResults(response);
      removeJob(jobId); // Remove from running jobs on success
      setCurrentJobId(null);
    } catch (err) {
      setError(err.message || 'Prediction failed');
      removeJob(jobId); // Remove from running jobs on error
      setCurrentJobId(null);
    } finally {
      setLoading(false);
    }
  };

  // ... rest of your component stays exactly the same ...

  return (
    <div className="space-y-6">
      {/* Input Method Tabs */}
      <div className="flex gap-2 border-b border-gray-200">
        <button
          onClick={() => setInputMethod('paste')}
          className={`px-6 py-3 font-medium transition-colors ${
            inputMethod === 'paste'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          <FileText className="w-4 h-4 inline mr-2" />
          Paste Sequence
        </button>
        <button
          onClick={() => setInputMethod('upload')}
          className={`px-6 py-3 font-medium transition-colors ${
            inputMethod === 'upload'
              ? 'text-blue-600 border-b-2 border-blue-600'
              : 'text-gray-600 hover:text-gray-900'
          }`}
        >
          <Upload className="w-4 h-4 inline mr-2" />
          Upload File
        </button>
      </div>

      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Paste Sequence */}
        {inputMethod === 'paste' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Paste your FASTA sequence:
            </label>
            <textarea
              value={sequence}
              onChange={(e) => setSequence(e.target.value)}
              placeholder={">genome_name\nATGCGATCGATCGATCG..."}
              rows={12}
              disabled={loading}
              className="w-full px-4 py-3 border border-gray-300 rounded-lg font-mono text-sm focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
            />
          </div>
        )}

        {/* Upload File */}
        {inputMethod === 'upload' && (
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              Upload FASTA file:
            </label>
            <div
              onDragOver={handleDragOver}
              onDrop={handleDrop}
              className={`border-2 border-dashed rounded-lg p-8 text-center transition-colors ${
                file
                  ? 'border-green-500 bg-green-50'
                  : 'border-gray-300 hover:border-blue-400 bg-gray-50'
              }`}
            >
              {!file ? (
                <div className="space-y-3">
                  <Upload className="w-12 h-12 mx-auto text-gray-400" />
                  <div>
                    <p className="text-gray-600">
                      Drag & drop your FASTA file here
                    </p>
                    <p className="text-sm text-gray-500 mt-1">or</p>
                  </div>
                  <label className="inline-block px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 cursor-pointer transition-colors">
                    <input
                      type="file"
                      accept=".fasta,.fa,.fna"
                      onChange={handleFileChange}
                      disabled={loading}
                      className="hidden"
                    />
                    Browse Files
                  </label>
                  <p className="text-xs text-gray-500">
                    Supported: .fasta, .fa, .fna
                  </p>
                </div>
              ) : (
                <div className="space-y-3">
                  <div className="w-12 h-12 mx-auto bg-green-100 rounded-full flex items-center justify-center">
                    <FileText className="w-6 h-6 text-green-600" />
                  </div>
                  <div>
                    <p className="font-medium text-gray-900">{file.name}</p>
                    <p className="text-sm text-gray-600">
                      {(file.size / 1024).toFixed(1)} KB
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setFile(null)}
                    disabled={loading}
                    className="text-sm text-red-600 hover:text-red-700"
                  >
                    Remove file
                  </button>
                </div>
              )}
            </div>
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
              Predicting...
            </>
          ) : (
            <>
              🧬 Predict Genes
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
      {results && (
        <div className="space-y-6 mt-8">
          <div className="bg-green-50 border border-green-200 rounded-lg p-6">
            <h2 className="text-xl font-bold text-gray-900 mb-4">
              ✅ Prediction Complete
            </h2>
            <div className="grid grid-cols-3 gap-4 text-center">
              <div>
                <p className="text-sm text-gray-600">Genome</p>
                <p className="text-lg font-semibold text-gray-900">
                  {results.genome_id}
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Sequence Length</p>
                <p className="text-lg font-semibold text-gray-900">
                  {results.sequence_length.toLocaleString()} bp
                </p>
              </div>
              <div>
                <p className="text-sm text-gray-600">Genes Found</p>
                <p className="text-lg font-semibold text-blue-600">
                  {results.total_genes}
                </p>
              </div>
            </div>
          </div>

          <div className="bg-white rounded-lg border border-gray-200">
            <div className="px-6 py-4 border-b border-gray-200">
              <h3 className="text-lg font-semibold text-gray-900">
                Predicted Genes
              </h3>
            </div>
            <div className="overflow-x-auto">
              <table className="w-full">
                <thead className="bg-gray-50">
                  <tr>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase">
                      Gene ID
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase">
                      Start
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase">
                      End
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase">
                      Length
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase">
                      Strand
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase">
                      Score
                    </th>
                    <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase">
                      RBS Score
                    </th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-200">
                  {results.predictions.map((gene) => (
                    <tr key={gene.gene_id} className="hover:bg-gray-50">
                      <td className="px-6 py-4 text-sm font-medium text-gray-900">
                        {gene.gene_id}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-700">
                        {gene.start.toLocaleString()}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-700">
                        {gene.end.toLocaleString()}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-700">
                        {gene.length.toLocaleString()} bp
                      </td>
                      <td className="px-6 py-4 text-sm">
                        {gene.strand === 'forward' ? (
                          <span className="text-blue-600">→ Forward</span>
                        ) : (
                          <span className="text-purple-600">← Reverse</span>
                        )}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-700">
                        {gene.combined_score.toFixed(3)}
                      </td>
                      <td className="px-6 py-4 text-sm text-gray-700">
                        {gene.rbs_score ? gene.rbs_score.toFixed(2) : 'N/A'}
                      </td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FastaMode;