import { useState } from 'react';
import { Download, Eye, Table2 } from 'lucide-react';
import GenomeViewer from './GenomeViewer';
import ResultsTable from './ResultsTable';

const ResultsView = ({ results, mode }) => {
  const [view, setView] = useState('visual');
  const [highlightedGeneId, setHighlightedGeneId] = useState(null);

  const handleGeneClick = (geneId) => {
    // Toggle behavior: if clicking same gene, deselect it
    if (geneId === highlightedGeneId) {
      setHighlightedGeneId(null);
      return;
    }
    
    setHighlightedGeneId(geneId);
    
    if (geneId) {
      setView('visual'); // Switch to visual view when selecting
      
      // Scroll to top of results to see the genome viewer
      setTimeout(() => {
        const resultsElement = document.querySelector('.genome-viewer-container');
        if (resultsElement) {
          resultsElement.scrollIntoView({ behavior: 'smooth', block: 'start' });
        }
      }, 100);
    }
  };

  const handleDownloadGFF = () => {
    let gffContent = "##gff-version 3\n";
    
    results.predictions.forEach(gene => {
      const strand = gene.strand === 'forward' ? '+' : '-';
      const attrs = `ID=${gene.gene_id};rbs_score=${gene.rbs_score || 0};combined_score=${gene.combined_score}`;
      gffContent += `${results.genome_id}\tHybridPredictor\tCDS\t${gene.start}\t${gene.end}\t${gene.combined_score.toFixed(3)}\t${strand}\t0\t${attrs}\n`;
    });

    const blob = new Blob([gffContent], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${results.genome_id}_predictions.gff`;
    a.click();
    URL.revokeObjectURL(url);
  };

  const handleDownloadCSV = () => {
    let csvContent = "Gene ID,Start,End,Length,Strand,Combined Score,RBS Score\n";
    
    results.predictions.forEach(gene => {
      csvContent += `${gene.gene_id},${gene.start},${gene.end},${gene.length},${gene.strand},${gene.combined_score},${gene.rbs_score || 'N/A'}\n`;
    });

    const blob = new Blob([csvContent], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${results.genome_id}_predictions.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  return (
    <div className="space-y-6">
      {/* Summary Card */}
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

      {/* View Controls */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          <button
            onClick={() => setView('visual')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              view === 'visual'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <Eye className="w-4 h-4" />
            Visual View
          </button>
          <button
            onClick={() => setView('table')}
            className={`flex items-center gap-2 px-4 py-2 rounded-lg font-medium transition-colors ${
              view === 'table'
                ? 'bg-blue-600 text-white'
                : 'bg-gray-100 text-gray-700 hover:bg-gray-200'
            }`}
          >
            <Table2 className="w-4 h-4" />
            Table View
          </button>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleDownloadGFF}
            className="flex items-center gap-2 px-4 py-2 bg-green-600 text-white rounded-lg hover:bg-green-700 transition-colors"
          >
            <Download className="w-4 h-4" />
            Download GFF
          </button>
          <button
            onClick={handleDownloadCSV}
            className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
          >
            <Download className="w-4 h-4" />
            Download CSV
          </button>
        </div>
      </div>

      {/* Content */}
      <div className="genome-viewer-container">
        {view === 'visual' ? (
          <GenomeViewer
            predictions={results.predictions}
            genomeLength={results.sequence_length}
            genomeId={results.genome_id}
            highlightGeneId={highlightedGeneId}
            onGeneClick={handleGeneClick}
          />
        ) : (
          <ResultsTable 
            predictions={results.predictions}
            onGeneClick={handleGeneClick}
            highlightedGeneId={highlightedGeneId}
          />
        )}
      </div>
    </div>
  );
};

export default ResultsView;