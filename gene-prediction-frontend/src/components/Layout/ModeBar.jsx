import { MODES } from '../PipelineVisualization/pipelineData';

const ModeBar = ({ activeMode, onModeChange }) => {
  return (
    <div className="bg-white border-b border-gray-200">
      <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
        <p className="text-sm font-semibold text-gray-600 mb-3">
          Choose Analysis Mode:
        </p>
        <div className="grid grid-cols-2 md:grid-cols-4 gap-3">
          {MODES.map(mode => {
            const Icon = mode.icon;
            const isActive = activeMode === mode.id;
            
            return (
              <button
                key={mode.id}
                onClick={() => onModeChange(isActive ? null : mode.id)}
                className={`p-4 rounded-lg border-2 transition-all text-left ${
                  isActive
                    ? 'border-blue-500 bg-blue-50 shadow-sm'
                    : 'border-gray-200 hover:border-blue-300 hover:bg-gray-50'
                }`}
              >
                <Icon 
                  className={`w-6 h-6 mb-2 ${
                    isActive ? 'text-blue-600' : 'text-gray-600'
                  }`}
                />
                <div className="text-sm font-semibold text-gray-900">
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
    </div>
  );
};

export default ModeBar;