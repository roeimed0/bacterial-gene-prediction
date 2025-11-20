import { useState } from 'react';
import { Filter, ChevronDown, ChevronUp } from 'lucide-react';
import PipelineStep from './PipelineStep';

const PipelineSteps = ({ steps, activeStep, onStepToggle, showML, activeMode }) => {
  // Add state to track if collapsed
  const [isCollapsed, setIsCollapsed] = useState(false);
  
  // Filter out ML steps if ML is disabled
  const visibleSteps = steps.filter(step => {
    if(step.ml && !showML) return false;

    if (step.alwaysShow && activeMode !== 'validate') return true;

    if (activeMode === null) return false;
  

    const matchesMode = 
    (step.needsCatalog && activeMode === 'catalog') ||
    (step.needsNcbi && activeMode === 'ncbi') ||
    (step.needsFasta && activeMode === 'fasta') ||
    (step.needsValidate && activeMode === 'validate');
  
    return matchesMode;
  });

  return (
    <div className="bg-white rounded-lg shadow-lg p-6">
      {/* Make the header clickable */}
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="w-full flex items-center justify-between mb-4 hover:bg-gray-50 p-2 rounded-lg transition-colors"
      >
        <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
          Prediction Pipeline
        </h2>
        {/* Show up/down arrow based on state */}
        {isCollapsed ? (
          <ChevronDown className="w-5 h-5 text-gray-600" />
        ) : (
          <ChevronUp className="w-5 h-5 text-gray-600" />
        )}
      </button>
      
      {/* Only show steps if NOT collapsed */}
      {!isCollapsed && (
        <div className="space-y-2">
          {visibleSteps.map((step,index) => (
            <PipelineStep
              key={step.name}
              step={step}
              index={index+1}
              isActive={activeStep === step.name}
              onToggle={() => onStepToggle(step.name)}
            />
          ))}
        </div>
      )}

      {!isCollapsed && (
        <div className="mt-4 pt-4 border-t border-gray-200">
          <p className="text-xs font-semibold text-gray-600 mb-2">Step Categories:</p>
          <div className="grid grid-cols-2 md:grid-cols-3 gap-2 text-xs">
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-blue-500"></div>
              <span className="text-gray-600">Input/Output</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-purple-500"></div>
              <span className="text-gray-600">Training</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-green-500"></div>
              <span className="text-gray-600">Filtering</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-yellow-500"></div>
              <span className="text-gray-600">Organization</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-red-500"></div>
              <span className="text-gray-600">Machine Learning</span>
            </div>
            <div className="flex items-center gap-2">
              <div className="w-3 h-3 rounded-full bg-orange-500"></div>
              <span className="text-gray-600">Validation</span>
            </div>
          </div>
        </div>
      )}

      
    </div>
  );
};

export default PipelineSteps;