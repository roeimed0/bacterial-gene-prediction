import { useState } from 'react';
import ModeSelector from './ModeSelector';
import MLToggle from './MLToggle';
import PipelineSteps from './PipelineSteps';
import KeyFeatures from './KeyFeatures';
import { MODES, PIPELINE_STEPS, ML_OPTIONS, FEATURES } from './pipelineData';

const PipelineVisualization = ({ onClose, initialMode = null, initialMLState = true }) => {
  const [activeMode, setActiveMode] = useState(initialMode);
  const [activeStep, setActiveStep] = useState(null);
  const [showML, setShowML] = useState(initialMLState);

  const handleStepToggle = (stepId) => {
    setActiveStep(activeStep === stepId ? null : stepId);
  };

  const handleMLToggle = () => {
    setShowML(prev => !prev);
  };

  return (
    <div className="w-full max-w-6xl mx-auto p-6 bg-gradient-to-br from-slate-50 to-blue-50">
      {/* Header with close button */}
      <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
        <div className="flex items-start justify-between">
          <div>
            <h1 className="text-3xl font-bold text-gray-800 mb-2">
              Hybrid Bacterial Gene Predictor
            </h1>
            <p className="text-gray-600">
              Traditional methods + Machine Learning pipeline
            </p>
          </div>
          {onClose && (
            <button
              onClick={onClose}
              className="text-gray-400 hover:text-gray-600 transition-colors"
              aria-label="Close pipeline visualization"
            >
              <svg 
                className="w-6 h-6" 
                fill="none" 
                stroke="currentColor" 
                viewBox="0 0 24 24"
                aria-hidden="true"
              >
                <path 
                  strokeLinecap="round" 
                  strokeLinejoin="round" 
                  strokeWidth={2} 
                  d="M6 18L18 6M6 6l12 12" 
                />
              </svg>
            </button>
          )}
        </div>
      </div>

      <ModeSelector 
        modes={MODES}
        activeMode={activeMode}
        onModeChange={setActiveMode}
      />

      <MLToggle 
        mlOptions={ML_OPTIONS}
        showML={showML}
        onToggle={handleMLToggle}
      />

      <PipelineSteps 
        steps={PIPELINE_STEPS}
        activeStep={activeStep}
        onStepToggle={handleStepToggle}
        showML={showML}
        activeMode={activeMode}
      />

      <KeyFeatures features={FEATURES} />
    </div>
  );
};

export default PipelineVisualization;