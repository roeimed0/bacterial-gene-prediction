import { useState } from 'react'
import PipelineVisualization from './components/PipelineVisualization'
import Header from './components/Layout/Header'
import ModeBar from './components/Layout/ModeBar'
import MainContent from './components/Layout/MainContent'

function App() {
  const [showPipeline, setShowPipeline] = useState(false)
  const [activeMode, setActiveMode] = useState(null)

  return (
    <div className="min-h-screen bg-gray-50">
      {/* Is this line here? */}
      <Header onShowPipeline={() => setShowPipeline(true)} />
      
      <ModeBar 
        activeMode={activeMode}
        onModeChange={setActiveMode}
      />
      
      <MainContent activeMode={activeMode} />
      
      {/* Modal - Is this still here? */}
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
                initialMode={activeMode}
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