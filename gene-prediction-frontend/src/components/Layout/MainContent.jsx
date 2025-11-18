import FastaMode from '../Modes/FastaMode';
import NcbiMode from '../Modes/NcbiMode';  // ADD THIS

const MainContent = ({ 
  activeMode, 
  fastaResults, 
  setFastaResults,
  catalogResults,
  setCatalogResults,
  ncbiResults,
  setNcbiResults,
  validateResults,
  setValidateResults,
  addJob,
  removeJob
}) => {
  return (
    <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
      {activeMode === null && (
        <div className="text-center py-16">
          <h2 className="text-3xl font-bold text-gray-900 mb-4">
            Welcome to Bacterial Gene Predictor
          </h2>
          <p className="text-lg text-gray-600 mb-8 max-w-2xl mx-auto">
            Choose an analysis mode above to get started.
          </p>
        </div>
      )}
      
      {activeMode === 'catalog' && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Browse Genome Catalog
          </h2>
          <p className="text-gray-600">
            Catalog content will go here...
          </p>
        </div>
      )}
      
      {/* NCBI MODE - UPDATE THIS SECTION */}
      {activeMode === 'ncbi' && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            Download from NCBI
          </h2>
          <NcbiMode 
            results={ncbiResults}
            setResults={setNcbiResults}
            addJob={addJob}
            removeJob={removeJob}
          />
        </div>
      )}
      
      {activeMode === 'fasta' && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-6">
            Upload Your FASTA File
          </h2>
          <FastaMode 
            results={fastaResults}
            setResults={setFastaResults}
            addJob={addJob}
            removeJob={removeJob}
          />
        </div>
      )}
      
      {activeMode === 'validate' && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Validate Predictions
          </h2>
          <p className="text-gray-600">
            Validation content will go here...
          </p>
        </div>
      )}
    </main>
  );
};

export default MainContent;