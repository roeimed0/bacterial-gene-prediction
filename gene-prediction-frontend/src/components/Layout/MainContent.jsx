const MainContent = ({ activeMode }) => {
  return (
    <main className="max-w-7xl mx-auto px-4 py-8 sm:px-6 lg:px-8">
      {/* No mode selected - Welcome screen */}
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
      
      {/* Catalog mode selected */}
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
      
      {/* NCBI mode selected */}
      {activeMode === 'ncbi' && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Download from NCBI
          </h2>
          <p className="text-gray-600">
            NCBI content will go here...
          </p>
        </div>
      )}
      
      {/* FASTA mode selected */}
      {activeMode === 'fasta' && (
        <div className="bg-white rounded-lg shadow-sm border border-gray-200 p-8">
          <h2 className="text-2xl font-bold text-gray-900 mb-4">
            Upload Your FASTA File
          </h2>
          <p className="text-gray-600">
            File upload will go here...
          </p>
        </div>
      )}
      
      {/* Validate mode selected */}
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