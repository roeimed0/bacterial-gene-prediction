import { useState } from 'react'
import PipelineVisualization from './components/PipelineVisualization'

function App() {
  const [showPipeline, setShowPipeline] = useState(false)

  return (
    <div className="min-h-screen bg-gray-100 relative">
      <button 
        onClick={() => setShowPipeline(true)}
        className="absolute top-8 right-8 bg-blue-600 text-white px-6 py-3 rounded-lg hover:bg-blue-700 transition-colors z-10"
      >
        Introduction
      </button>
      
      <div className="min-h-screen flex items-center justify-center">
        <h1 className="text-4xl font-bold text-blue-600">
          Bacterial and Archeal Gene Predictor
        </h1>
      </div>

      {/* Modal Overlay - Scrollable */}
      {showPipeline && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-50 overflow-y-auto"
          onClick={() => setShowPipeline(false)}
        >
          <div className="min-h-screen flex items-center justify-center p-4">
            <div 
              className="w-full max-w-6xl"
              onClick={(e) => e.stopPropagation()}
            >
              <PipelineVisualization 
                onClose={() => setShowPipeline(false)}
                initialMLState={true}
              />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App