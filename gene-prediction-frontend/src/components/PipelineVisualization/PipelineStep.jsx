import { ChevronDown, ChevronRight } from 'lucide-react';

const PipelineStep = ({ step,index, isActive, onToggle }) => {
  return (
    <div>
      <button
        onClick={onToggle}
        className="w-full flex items-center gap-3 p-4 rounded-lg hover:bg-gray-50 transition-all"
        aria-expanded={isActive}
        aria-label={`${step.name}: ${step.desc}`}
      >
        <div 
          className={`w-12 h-12 rounded-full ${step.color} flex items-center justify-center text-white font-bold flex-shrink-0`}
          aria-hidden="true"
        >
          {index}
        </div>
        <div className="flex-1 text-left">
          <div className="font-semibold text-gray-800">{step.name}</div>
          <div className="text-sm text-gray-600">{step.desc}</div>
        </div>
        {step.ml && (
          <span 
            className="px-3 py-1 bg-red-100 text-red-700 rounded-full text-xs font-semibold"
            aria-label="Machine learning step"
          >
            ML
          </span>
        )}
        {isActive ? (
          <ChevronDown className="w-5 h-5 text-gray-400" aria-hidden="true" />
        ) : (
          <ChevronRight className="w-5 h-5 text-gray-400" aria-hidden="true" />
        )}
      </button>
      
      {isActive && (
        <div className="ml-16 mr-4 mb-4 p-4 bg-gray-50 rounded-lg border-l-4 border-gray-300">
          <p className="text-sm text-gray-700 mb-2">{step.details}</p>
          {step.note && (
            <div className="mt-2 p-2 bg-yellow-50 border border-yellow-200 rounded">
              <strong>Note:</strong> {step.note}
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default PipelineStep;