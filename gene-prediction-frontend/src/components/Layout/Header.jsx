import { Info } from 'lucide-react';
import ProcessManager from '../ProcessManager';

const Header = ({ onShowPipeline, runningJobs, onRemoveJob }) => {
  return (
    <header className="bg-white border-b border-gray-200 shadow-sm">
      <div className="max-w-7xl mx-auto px-4 py-4 sm:px-6 lg:px-8">
        <div className="flex items-center justify-between">
          {/* Left side: Logo + Title */}
          <div className="flex items-center gap-3">
            <div>
              <h1 className="text-2xl font-bold text-gray-900">
                Bacterial and Archeal Gene Predictor
              </h1>
              <p className="text-sm text-gray-500">
                Hybrid ML-powered prediction pipeline
              </p>
            </div>
          </div>
          
          {/* Right side: Process Manager + Introduction button */}
          <div className="flex items-center gap-3">
            {/* Process Manager */}
            <ProcessManager 
              runningJobs={runningJobs}
              onRemove={onRemoveJob}
            />
            
            {/* Introduction button */}
            <button
              onClick={onShowPipeline}
              className="flex items-center gap-2 px-4 py-2 bg-blue-600 text-white rounded-lg hover:bg-blue-700 transition-colors"
            >
              <Info className="w-4 h-4" />
              <span>Introduction</span>
            </button>
          </div>
        </div>
      </div>
    </header>
  );
};

export default Header;