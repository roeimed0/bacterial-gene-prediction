import { useState, useEffect } from 'react';
import { Search, Loader, Download } from 'lucide-react';
import { api } from '../../services/api';
import ResultsView from '../Results/ResultsView';

const CatalogMode = ({ results, setResults, addJob, removeJob }) => {
  const [catalog, setCatalog] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [searchTerm, setSearchTerm] = useState('');
  const [selectedGenome, setSelectedGenome] = useState(null);
  const [predicting, setPredicting] = useState(false);

  // ML Options
  const [useGroupML, setUseGroupML] = useState(true);
  const [useFinalML, setUseFinalML] = useState(true);
  const [groupThreshold, setGroupThreshold] = useState(0.1);
  const [finalThreshold, setFinalThreshold] = useState(0.12);

  // Load catalog on mount
  useEffect(() => {
    loadCatalog();
  }, []);

  const loadCatalog = async () => {
    try {
      setLoading(true);
      const data = await api.getCatalog();
      setCatalog(data);
    } catch (err) {
      setError(err.message || 'Failed to load catalog');
    } finally {
      setLoading(false);
    }
  };

  const handlePredict = async (genome) => {
    setSelectedGenome(genome);
    setResults(null);
    setPredicting(true);

    const options = {
      useGroupML,
      groupThreshold,
      useFinalML,
      finalThreshold
    };

    // Add job to tracker
    const jobId = addJob({
      name: `${genome.accession} - ${genome.name}`,
      mode: 'Catalog',
      type: 'prediction'
    });

    try {
      // Use NCBI endpoint with a default email
      const response = await api.predictFromNcbi(
        genome.accession,
        'catalog@genomepredictor.com',
        options
      );
      setResults(response);
      removeJob(jobId);
    } catch (err) {
      setError(err.message || 'Prediction failed');
      removeJob(jobId);
    } finally {
      setPredicting(false);
    }
  };

  // Group genomes by taxonomic group
  const groupedGenomes = catalog?.genomes.reduce((groups, genome) => {
    const group = genome.group;
    if (!groups[group]) {
      groups[group] = [];
    }
    groups[group].push(genome);
    return groups;
  }, {});

  // Filter genomes by search term
  const filterGenomes = (genomes) => {
    if (!searchTerm) return genomes;
    const term = searchTerm.toLowerCase();
    return genomes.filter(g => 
      g.name.toLowerCase().includes(term) || 
      g.accession.toLowerCase().includes(term) ||
      g.id.toString().includes(term)
    );
  };

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader className="w-8 h-8 animate-spin text-blue-600" />
        <span className="ml-3 text-gray-600">Loading catalog...</span>
      </div>
    );
  }

  if (error && !catalog) {
    return (
      <div className="bg-red-50 border border-red-200 rounded-lg p-4">
        <p className="text-red-800 font-medium">❌ {error}</p>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Search Bar */}
      <div className="flex items-center gap-4">
        <div className="flex-1 relative">
          <Search className="absolute left-3 top-1/2 transform -translate-y-1/2 w-5 h-5 text-gray-400" />
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            placeholder="Search by name, accession, or number..."
            className="w-full pl-10 pr-4 py-2 border border-gray-300 rounded-lg focus:ring-2 focus:ring-blue-500 focus:border-transparent"
          />
        </div>
        <div className="text-sm text-gray-600">
          {catalog?.total} genomes
        </div>
      </div>

      {/* ML Options */}
      <div className="bg-gray-50 rounded-lg p-4 space-y-3">
        <h3 className="font-semibold text-gray-900 text-sm">
          Machine Learning Options
        </h3>
        <div className="grid grid-cols-2 gap-4">
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useGroupML}
              onChange={(e) => setUseGroupML(e.target.checked)}
              disabled={predicting}
              className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700">ML Group Filtering</span>
          </label>
          <label className="flex items-center gap-2 cursor-pointer">
            <input
              type="checkbox"
              checked={useFinalML}
              onChange={(e) => setUseFinalML(e.target.checked)}
              disabled={predicting}
              className="w-4 h-4 text-blue-600 rounded focus:ring-blue-500"
            />
            <span className="text-sm text-gray-700">Final ML Filtration</span>
          </label>
        </div>
      </div>

      {/* 4-Column Grid Layout */}
      <div className="grid grid-cols-4 gap-4">
        {groupedGenomes && Object.keys(groupedGenomes).map((groupName) => {
          const genomes = filterGenomes(groupedGenomes[groupName]);
          
          return (
            <div key={groupName} className="bg-white rounded-lg border border-gray-200 overflow-hidden">
              {/* Group Header */}
              <div className="bg-blue-600 text-white px-4 py-3">
                <h3 className="font-bold text-sm">{groupName}</h3>
                <p className="text-xs opacity-90">{genomes.length} genomes</p>
              </div>
              
              {/* Genome List */}
              <div className="divide-y divide-gray-100 max-h-[600px] overflow-y-auto">
                {genomes.map((genome) => (
                  <div
                    key={genome.id}
                    className="p-3 hover:bg-blue-50 transition-colors cursor-pointer group"
                    onClick={() => handlePredict(genome)}
                  >
                    <div className="flex items-start justify-between gap-2">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center gap-2 mb-1">
                          <span className="text-xs font-semibold text-gray-500">
                            #{genome.id}
                          </span>
                          <span className="text-xs font-mono text-blue-600">
                            {genome.accession}
                          </span>
                        </div>
                        <p className="text-sm text-gray-900 line-clamp-2 group-hover:text-blue-700">
                          {genome.name}
                        </p>
                      </div>
                      <Download className="w-4 h-4 text-gray-400 group-hover:text-blue-600 flex-shrink-0 mt-1" />
                    </div>
                  </div>
                ))}
              </div>
            </div>
          );
        })}
      </div>

      {/* No Results */}
      {searchTerm && Object.values(groupedGenomes || {}).every(g => filterGenomes(g).length === 0) && (
        <div className="text-center py-8 text-gray-500">
          No genomes found matching "{searchTerm}"
        </div>
      )}

      {/* Loading Prediction */}
      {predicting && selectedGenome && (
        <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
          <div className="flex items-center gap-3">
            <Loader className="w-5 h-5 animate-spin text-blue-600" />
            <div>
              <p className="font-medium text-gray-900">
                Downloading and predicting...
              </p>
              <p className="text-sm text-gray-600">
                {selectedGenome.accession} - {selectedGenome.name}
              </p>
            </div>
          </div>
        </div>
      )}

      {/* Error Display */}
      {error && catalog && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800 font-medium">❌ {error}</p>
        </div>
      )}

      {/* Results Display */}
      {results && <ResultsView results={results} mode="catalog" />}
    </div>
  );
};

export default CatalogMode;