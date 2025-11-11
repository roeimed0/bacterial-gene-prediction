import { Settings } from 'lucide-react';

const ModeSelector = ({ modes, activeMode, onModeChange }) => {
  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
      <h2 className="text-xl font-bold text-gray-800 mb-4 flex items-center gap-2">
        Operating Modes
      </h2>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
        {modes.map(mode => {
          const Icon = mode.icon;
          const isActive = activeMode === mode.id;
          
          return (
            <button
              key={mode.id}
              onClick={() => onModeChange(activeMode===mode.id ? null : mode.id)}
              className={`p-4 rounded-lg border-2 transition-all ${
                isActives
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-blue-300'
              }`}
              aria-pressed={isActive}
              aria-label={`Select ${mode.name} mode`}
            >
              <Icon 
                className={`w-6 h-6 mx-auto mb-2 ${
                  isActive ? 'text-blue-600' : 'text-gray-600'
                }`}
                aria-hidden="true"
              />
              <div className="text-sm font-semibold text-gray-800">
                {mode.name}
              </div>
              <div className="text-xs text-gray-500 mt-1">
                {mode.desc}
              </div>
            </button>
          );
        })}
      </div>
    </div>
  );
};

export default ModeSelector;