import { useState } from 'react';
import { Download, Loader } from 'lucide-react';
import { api } from '../../services/api';

const NcbiMode = ({ results, setResults, addJob, removeJob }) => {
  const [accession, setAccession] = useState('');
  const [email, setEmail] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState(null);
  const [currentJobId, setCurrentJobId] = useState(null);

  // ML Options
  const [useGroupML, setUseGroupML] = useState(true);
  const [useFinalML, setUseFinalML] = useState(true);
  const [groupThreshold, setGroupThreshold] = useState(0.1);
  const [finalThreshold, setFinalThreshold] = useState(0.12);

  const validateAccession = (acc) => {
    // NCBI accession format: XX_XXXXXX.X (e.g., NC_000913.3)
    const pattern = /^[A-Z]{2}_\d{6,}\.\d+$/;
    return pattern.test(acc);
  };

  const validateEmail = (em) => {
    return em.includes('@') && em.includes('.');
  };

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError(null);
    setResults(null);

    // Validation
    if (!accession.trim()) {
      setError('Please enter an NCBI accession number');
      return;
    }

    if (!validateAccession(accession)) {
      setError('Invalid accession format. Example: NC_000913.3');
      return;
    }

    if (!email.trim()) {
      setError('Email is required by NCBI');
      return;
    }

    if (!validateEmail(email)) {
      setError('Please enter a valid email address');
      return;
    }

    setLoading(true);

    const options = {
      useGroupML,
      groupThreshold,
      useFinalML,
      finalThreshold
    };

    // Add job to tracker
    const jobId = addJob({
      name: accession,
      mode: 'NCBI Download',
      type: 'prediction'
    });
    setCurrentJobId(jobId);

    try {
      const response = await api.predictFromNcbi(accession, email, options);
      setResults(response);
      removeJob(jobId);
      setCurrentJobId(null);
    } catch (err) {
      setError(err.message || 'NCBI download failed');
      removeJob(jobId);
      setCurrentJobId(null);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="space-y-6">
      <form onSubmit={handleSubmit} className="space-y-6">
        {/* Accession Input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            NCBI Accession Number:
          </label>
          <input
            type="text"
            value={accession}
            onChange={(e) => setAccession(e.target.value.trim())}
            placeholder="NC_000913.3"
            disabled={loading}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
          />
          <p className="text-xs text-gray-500 mt-1">
            Format: XX_XXXXXX.X (e.g., NC_000913.3 for E. coli K-12)
          </p>
        </div>

        {/* Email Input */}
        <div>
          <label className="block text-sm font-medium text-gray-700 mb-2">
            Email Address (required by NCBI):
          </label>
          <input
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value.trim())}
            placeholder="your@email.com"
            disabled={loading}
            className="w-full px-4 py-3 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent disabled:bg-gray-100"
          />
          <p className="text-xs text-gray-500 mt-1">
            NCBI requires an email for API access compliance
          </p>
        </div>

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
              Downloading & Predicting...
            </>
          ) : (
            <>
              <Download className="w-5 h-5" />
              Download & Predict
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
                <p className="text-sm text-gray-600">Accession</p>
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

export default NcbiMode;