import { useState } from 'react'
import PipelineVisualization from './components/PipelineVisualization'
import Header from './components/Layout/Header'
import ModeBar from './components/Layout/ModeBar'
import MainContent from './components/Layout/MainContent'
import FileManager from './components/FileManager'  // ADD THIS

function App() {
  const [showPipeline, setShowPipeline] = useState(false)
  const [showFileManager, setShowFileManager] = useState(false)  // ADD THIS
  const [activeMode, setActiveMode] = useState(null)
  
  // Results state
  const [fastaResults, setFastaResults] = useState(null)
  const [catalogResults, setCatalogResults] = useState(null)
  const [ncbiResults, setNcbiResults] = useState(null)
  const [validateResults, setValidateResults] = useState(null)
  
  // Running jobs tracker
  const [runningJobs, setRunningJobs] = useState([])
  
  const addJob = (jobData) => {
    const job = {
      id: Date.now(),
      ...jobData,
      startTime: new Date()
    }
    setRunningJobs(prev => [...prev, job])
    return job.id
  }
  
  const removeJob = (jobId) => {
    setRunningJobs(prev => prev.filter(job => job.id !== jobId))
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header 
        onShowPipeline={() => setShowPipeline(true)}
        onShowFileManager={() => setShowFileManager(true)}  // ADD THIS
        runningJobs={runningJobs}
        onRemoveJob={removeJob}
      />
      
      <ModeBar 
        activeMode={activeMode}
        onModeChange={setActiveMode}
      />
      
      <MainContent 
        activeMode={activeMode}
        fastaResults={fastaResults}
        setFastaResults={setFastaResults}
        catalogResults={catalogResults}
        setCatalogResults={setCatalogResults}
        ncbiResults={ncbiResults}
        setNcbiResults={setNcbiResults}
        validateResults={validateResults}
        setValidateResults={setValidateResults}
        addJob={addJob}
        removeJob={removeJob}
      />
      
      {/* Pipeline Visualization Modal */}
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

      {/* File Manager Modal - ADD THIS */}
      {showFileManager && (
        <div 
          className="fixed inset-0 bg-black bg-opacity-50 z-50 overflow-y-auto"
          onClick={() => setShowFileManager(false)}
        >
          <div className="min-h-screen flex items-center justify-center p-4">
            <div 
              className="w-full max-w-6xl bg-white rounded-lg shadow-xl p-6"
              onClick={(e) => e.stopPropagation()}
            >
              <div className="flex justify-between items-center mb-4">
                <h2 className="text-2xl font-bold text-gray-900">File Manager</h2>
                <button
                  onClick={() => setShowFileManager(false)}
                  className="text-gray-400 hover:text-gray-600 transition-colors"
                >
                  <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                  </svg>
                </button>
              </div>
              <FileManager />
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App