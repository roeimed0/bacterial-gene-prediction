import { useState } from 'react';
import { ChevronLeft, ChevronRight, ChevronsLeft, ChevronsRight, ArrowUpDown, MapPin } from 'lucide-react';

const ResultsTable = ({ predictions, onGeneClick, highlightedGeneId }) => {
  const [currentPage, setCurrentPage] = useState(1);
  const [itemsPerPage, setItemsPerPage] = useState(100);
  const [sortConfig, setSortConfig] = useState({ key: null, direction: 'asc' });

  // Sorting
  const sortedPredictions = [...predictions].sort((a, b) => {
    if (!sortConfig.key) return 0;

    let aVal = a[sortConfig.key];
    let bVal = b[sortConfig.key];

    if (sortConfig.key === 'strand') {
      aVal = a.strand === 'forward' ? 1 : 0;
      bVal = b.strand === 'forward' ? 1 : 0;
    }

    if (aVal < bVal) return sortConfig.direction === 'asc' ? -1 : 1;
    if (aVal > bVal) return sortConfig.direction === 'asc' ? 1 : -1;
    return 0;
  });

  // Pagination
  const totalPages = Math.ceil(sortedPredictions.length / itemsPerPage);
  const startIndex = (currentPage - 1) * itemsPerPage;
  const endIndex = startIndex + itemsPerPage;
  const currentPredictions = sortedPredictions.slice(startIndex, endIndex);

  const handleSort = (key) => {
    setSortConfig({
      key,
      direction: sortConfig.key === key && sortConfig.direction === 'asc' ? 'desc' : 'asc'
    });
  };

  const getSortIcon = (key) => {
    if (sortConfig.key !== key) return <ArrowUpDown className="w-4 h-4 text-gray-400" />;
    return sortConfig.direction === 'asc' ? '↑' : '↓';
  };

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-4">
          <span className="text-sm text-gray-600">Show:</span>
          <select
            value={itemsPerPage}
            onChange={(e) => {
              setItemsPerPage(Number(e.target.value));
              setCurrentPage(1);
            }}
            className="px-3 py-1 border border-gray-300 rounded-lg text-sm"
          >
            <option value={100}>100</option>
            <option value={500}>500</option>
            <option value={1000}>1000</option>
            <option value={predictions.length}>All ({predictions.length})</option>
          </select>
          <span className="text-sm text-gray-600">
            Showing {startIndex + 1}-{Math.min(endIndex, predictions.length)} of {predictions.length}
          </span>
        </div>

        {/* Pagination Controls */}
        <div className="flex items-center gap-2">
          <button
            onClick={() => setCurrentPage(1)}
            disabled={currentPage === 1}
            className="p-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronsLeft className="w-4 h-4" />
          </button>
          <button
            onClick={() => setCurrentPage(currentPage - 1)}
            disabled={currentPage === 1}
            className="p-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronLeft className="w-4 h-4" />
          </button>
          <span className="text-sm text-gray-600 min-w-[100px] text-center">
            Page {currentPage} of {totalPages}
          </span>
          <button
            onClick={() => setCurrentPage(currentPage + 1)}
            disabled={currentPage === totalPages}
            className="p-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronRight className="w-4 h-4" />
          </button>
          <button
            onClick={() => setCurrentPage(totalPages)}
            disabled={currentPage === totalPages}
            className="p-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
          >
            <ChevronsRight className="w-4 h-4" />
          </button>
        </div>
      </div>

      {/* Table */}
      <div className="bg-white rounded-lg border border-gray-200 overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full">
            <thead className="bg-gray-50">
              <tr>
                <th className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase">
                  View
                </th>
                <th 
                  onClick={() => handleSort('gene_id')}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase cursor-pointer hover:bg-gray-100"
                >
                  <div className="flex items-center gap-2">
                    Gene ID {getSortIcon('gene_id')}
                  </div>
                </th>
                <th 
                  onClick={() => handleSort('start')}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase cursor-pointer hover:bg-gray-100"
                >
                  <div className="flex items-center gap-2">
                    Start {getSortIcon('start')}
                  </div>
                </th>
                <th 
                  onClick={() => handleSort('end')}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase cursor-pointer hover:bg-gray-100"
                >
                  <div className="flex items-center gap-2">
                    End {getSortIcon('end')}
                  </div>
                </th>
                <th 
                  onClick={() => handleSort('length')}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase cursor-pointer hover:bg-gray-100"
                >
                  <div className="flex items-center gap-2">
                    Length {getSortIcon('length')}
                  </div>
                </th>
                <th 
                  onClick={() => handleSort('strand')}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase cursor-pointer hover:bg-gray-100"
                >
                  <div className="flex items-center gap-2">
                    Strand {getSortIcon('strand')}
                  </div>
                </th>
                <th 
                  onClick={() => handleSort('combined_score')}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase cursor-pointer hover:bg-gray-100"
                >
                  <div className="flex items-center gap-2">
                    Score {getSortIcon('combined_score')}
                  </div>
                </th>
                <th 
                  onClick={() => handleSort('rbs_score')}
                  className="px-6 py-3 text-left text-xs font-medium text-gray-700 uppercase cursor-pointer hover:bg-gray-100"
                >
                  <div className="flex items-center gap-2">
                    RBS Score {getSortIcon('rbs_score')}
                  </div>
                </th>
              </tr>
            </thead>
            <tbody className="divide-y divide-gray-200">
              {currentPredictions.map((gene) => (
                <tr 
                  key={gene.gene_id} 
                  className={`hover:bg-gray-50 ${
                    highlightedGeneId === gene.gene_id ? 'bg-amber-50' : ''
                  }`}
                >
                  <td className="px-6 py-4 text-sm">
                    <button
                      onClick={() => onGeneClick(gene.gene_id)}
                      className="p-1 text-blue-600 hover:text-blue-800 hover:bg-blue-100 rounded"
                      title="View in genome browser"
                    >
                      <MapPin className="w-4 h-4" />
                    </button>
                  </td>
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
  );
};

export default ResultsTable;