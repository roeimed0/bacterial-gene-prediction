import { useState } from 'react';
import { X, ChevronDown, ChevronUp, Loader } from 'lucide-react';

const ProcessManager = ({ runningJobs, onRemove }) => {
  const [isCollapsed, setIsCollapsed] = useState(false);

  if (runningJobs.length === 0) return null;

  return (
    <div className="relative">
      <button
        onClick={() => setIsCollapsed(!isCollapsed)}
        className="flex items-center gap-2 px-4 py-2 bg-blue-100 text-blue-700 rounded-lg hover:bg-blue-200 transition-colors"
      >
        <Loader className="w-4 h-4 animate-spin" />
        <span className="font-medium">
          Running Jobs ({runningJobs.length})
        </span>
        {isCollapsed ? (
          <ChevronDown className="w-4 h-4" />
        ) : (
          <ChevronUp className="w-4 h-4" />
        )}
      </button>

      {/* Dropdown Panel */}
      {!isCollapsed && (
        <div className="absolute right-0 mt-2 w-96 bg-white rounded-lg shadow-lg border border-gray-200 z-50">
          <div className="p-3 border-b border-gray-200 bg-gray-50">
            <h3 className="font-semibold text-gray-900">Active Predictions</h3>
          </div>
          <div className="max-h-80 overflow-y-auto">
            {runningJobs.map((job) => (
              <div 
                key={job.id}
                className="flex items-start justify-between p-4 border-b border-gray-100 hover:bg-gray-50 transition-colors"
              >
                <div className="flex-1 min-w-0 mr-3">
                  <div className="flex items-center gap-2 mb-1">
                    <Loader className="w-3 h-3 text-blue-600 animate-spin flex-shrink-0" />
                    <p className="text-sm font-medium text-gray-900 truncate">
                      {job.name}
                    </p>
                  </div>
                  <p className="text-xs text-gray-500">
                    {job.mode}
                  </p>
                  <p className="text-xs text-gray-400 mt-1">
                    Started at {new Date(job.startTime).toLocaleTimeString()}
                  </p>
                </div>
                <button
                  onClick={() => onRemove(job.id)}
                  className="p-1 text-gray-400 hover:text-red-600 hover:bg-red-50 rounded transition-colors flex-shrink-0"
                  title="Remove from list (backend will continue)"
                >
                  <X className="w-4 h-4" />
                </button>
              </div>
            ))}
          </div>
          <div className="p-3 bg-gray-50 border-t border-gray-200">
            <p className="text-xs text-gray-600">
              ℹ️ Removing from list doesn't stop the backend process
            </p>
          </div>
        </div>
      )}
    </div>
  );
};

export default ProcessManager;