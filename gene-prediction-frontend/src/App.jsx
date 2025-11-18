import { useState } from 'react'
import PipelineVisualization from './components/PipelineVisualization'
import Header from './components/Layout/Header'
import ModeBar from './components/Layout/ModeBar'
import MainContent from './components/Layout/MainContent'

function App() {
  const [showPipeline, setShowPipeline] = useState(false)
  const [activeMode, setActiveMode] = useState(null)
  
  // Results state
  const [fastaResults, setFastaResults] = useState(null)
  const [catalogResults, setCatalogResults] = useState(null)
  const [ncbiResults, setNcbiResults] = useState(null)
  const [validateResults, setValidateResults] = useState(null)
  
  // Running jobs tracker
  const [runningJobs, setRunningJobs] = useState([])
  
  // Add a new job
  const addJob = (jobData) => {
    const job = {
      id: Date.now(), // Simple unique ID
      ...jobData,
      startTime: new Date()
    }
    setRunningJobs(prev => [...prev, job])
    return job.id
  }
  
  // Remove a job (when completed or cancelled)
  const removeJob = (jobId) => {
    setRunningJobs(prev => prev.filter(job => job.id !== jobId))
  }

  return (
    <div className="min-h-screen bg-gray-50">
      <Header 
        onShowPipeline={() => setShowPipeline(true)}
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