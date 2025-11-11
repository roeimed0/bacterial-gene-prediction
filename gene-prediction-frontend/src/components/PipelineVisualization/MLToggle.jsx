import { Brain } from 'lucide-react';

const MLToggle = ({ mlOptions, showML, onToggle }) => {
  return (
    <div className="bg-white rounded-lg shadow-lg p-6 mb-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-gray-800 flex items-center gap-2">
          Machine Learning Options
        </h2>
        <button
          onClick={onToggle}
          className={`px-4 py-2 rounded-lg font-semibold transition-all ${
            showML
              ? 'bg-blue-500 text-white'
              : 'bg-gray-200 text-gray-600'
          }`}
          aria-pressed={showML}
          aria-label={`Machine learning is ${showML ? 'enabled' : 'disabled'}. Click to ${showML ? 'disable' : 'enable'}`}
        >
          {showML ? 'ML Enabled' : 'ML Disabled'}
        </button>
      </div>
      <div className="grid md:grid-cols-2 gap-4">
        {mlOptions.map(opt => (
          <div 
            key={opt.name} 
            className={`p-4 rounded-lg border-2 transition-colors ${
              showML ? 'border-blue-200 bg-blue-50' : 'border-gray-200 bg-gray-50'
            }`}
          >
            <div className="font-semibold text-gray-800">{opt.name}</div>
            <div className="text-sm text-gray-600 mt-1">{opt.desc}</div>
            <div className="text-xs text-gray-500 mt-2">
              <code className="bg-white px-2 py-1 rounded">
                {opt.param} {opt.default}
              </code>
            </div>
          </div>
        ))}
      </div>
    </div>
  );
};

export default MLToggle;