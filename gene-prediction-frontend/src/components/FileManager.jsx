import { useState, useEffect, useRef } from 'react';
import { Trash2, RefreshCw, Loader, AlertTriangle, File, FolderOpen } from 'lucide-react';
import { api } from '../services/api';

const FileManager = () => {
  const [files, setFiles] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);
  const [selectedFiles, setSelectedFiles] = useState(new Set());
  const [showConfirmDelete, setShowConfirmDelete] = useState(false);
  const [deleting, setDeleting] = useState(false);
  
  // Drag selection state
  const [isDragging, setIsDragging] = useState(false);
  const [selectionBox, setSelectionBox] = useState(null);
  const containerRef = useRef(null);
  const fileRefs = useRef({});

  useEffect(() => {
    loadFiles();
  }, []);

  const loadFiles = async () => {
    try {
      setLoading(true);
      const data = await api.getFiles();
      setFiles(data.files);
      setSelectedFiles(new Set());
    } catch (err) {
      setError(err.message || 'Failed to load files');
    } finally {
      setLoading(false);
    }
  };

  const toggleFileSelection = (path) => {
    const newSelected = new Set(selectedFiles);
    if (newSelected.has(path)) {
      newSelected.delete(path);
    } else {
      newSelected.add(path);
    }
    setSelectedFiles(newSelected);
  };

  const selectAll = () => {
    if (selectedFiles.size === files.length) {
      setSelectedFiles(new Set());
    } else {
      setSelectedFiles(new Set(files.map(f => f.path)));
    }
  };

  const handleDelete = async () => {
    if (selectedFiles.size === 0) return;

    setDeleting(true);
    try {
      const result = await api.deleteFiles(Array.from(selectedFiles));
      
      if (result.failed > 0) {
        setError(`Deleted ${result.deleted} files. Failed: ${result.failed}`);
      }
      
      await loadFiles();
      setShowConfirmDelete(false);
    } catch (err) {
      setError(err.message || 'Delete failed');
    } finally {
      setDeleting(false);
    }
  };

  const handleCleanupAll = async () => {
    setDeleting(true);
    try {
      const result = await api.cleanupAllFiles();
      
      if (result.failed > 0) {
        setError(`Deleted ${result.deleted} files. Failed: ${result.failed}`);
      }
      
      await loadFiles();
      setShowConfirmDelete(false);
    } catch (err) {
      setError(err.message || 'Cleanup failed');
    } finally {
      setDeleting(false);
    }
  };

  // Drag selection handlers
  const handleMouseDown = (e) => {
    if (e.target !== containerRef.current && !e.target.closest('.file-card')) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    setIsDragging(true);
    setSelectionBox({
      startX: e.clientX - rect.left,
      startY: e.clientY - rect.top,
      endX: e.clientX - rect.left,
      endY: e.clientY - rect.top
    });
  };

  const handleMouseMove = (e) => {
    if (!isDragging) return;
    
    const rect = containerRef.current.getBoundingClientRect();
    setSelectionBox(prev => ({
      ...prev,
      endX: e.clientX - rect.left,
      endY: e.clientY - rect.top
    }));

    // Check which files intersect with selection box
    const newSelected = new Set();
    Object.entries(fileRefs.current).forEach(([path, ref]) => {
      if (!ref) return;
      const fileRect = ref.getBoundingClientRect();
      const containerRect = containerRef.current.getBoundingClientRect();
      
      const fileBox = {
        left: fileRect.left - containerRect.left,
        top: fileRect.top - containerRect.top,
        right: fileRect.right - containerRect.left,
        bottom: fileRect.bottom - containerRect.top
      };

      const selBox = {
        left: Math.min(selectionBox.startX, selectionBox.endX),
        top: Math.min(selectionBox.startY, selectionBox.endY),
        right: Math.max(selectionBox.startX, selectionBox.endX),
        bottom: Math.max(selectionBox.startY, selectionBox.endY)
      };

      // Check intersection
      if (!(fileBox.right < selBox.left || 
            fileBox.left > selBox.right || 
            fileBox.bottom < selBox.top || 
            fileBox.top > selBox.bottom)) {
        newSelected.add(path);
      }
    });

    setSelectedFiles(newSelected);
  };

  const handleMouseUp = () => {
    setIsDragging(false);
    setSelectionBox(null);
  };

  useEffect(() => {
    if (isDragging) {
      window.addEventListener('mousemove', handleMouseMove);
      window.addEventListener('mouseup', handleMouseUp);
      return () => {
        window.removeEventListener('mousemove', handleMouseMove);
        window.removeEventListener('mouseup', handleMouseUp);
      };
    }
  }, [isDragging, selectionBox]);

  const formatSize = (bytes) => {
    if (bytes < 1024) return `${bytes} B`;
    if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
    return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  };

  const formatDate = (timestamp) => {
    const date = new Date(timestamp * 1000);
    return date.toLocaleString();
  };

  const getFileIcon = (type) => {
    switch (type) {
      case 'genome': return '🧬';
      case 'result': return '📊';
      case 'report': return '📄';
      default: return '📁';
    }
  };

  const totalSize = files.reduce((sum, f) => sum + f.size, 0);

  if (loading) {
    return (
      <div className="flex items-center justify-center py-12">
        <Loader className="w-8 h-8 animate-spin text-blue-600" />
        <span className="ml-3 text-gray-600">Loading files...</span>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-2xl font-bold text-gray-900">File Manager</h2>
          <p className="text-sm text-gray-600 mt-1">
            {files.length} files • {formatSize(totalSize)} total
          </p>
        </div>
        <div className="flex gap-2">
          <button
            onClick={loadFiles}
            disabled={loading}
            className="flex items-center gap-2 px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
          >
            <RefreshCw className={`w-4 h-4 ${loading ? 'animate-spin' : ''}`} />
            Refresh
          </button>
          <button
            onClick={selectAll}
            className="px-4 py-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-50"
          >
            {selectedFiles.size === files.length ? 'Deselect All' : 'Select All'}
          </button>
          <button
            onClick={() => setShowConfirmDelete(true)}
            disabled={selectedFiles.size === 0}
            className="flex items-center gap-2 px-4 py-2 text-white bg-red-600 rounded-lg hover:bg-red-700 disabled:bg-gray-400 disabled:cursor-not-allowed"
          >
            <Trash2 className="w-4 h-4" />
            Delete Selected ({selectedFiles.size})
          </button>
        </div>
      </div>

      {/* Instructions */}
      <div className="bg-blue-50 border border-blue-200 rounded-lg p-4">
        <p className="text-sm text-blue-900">
          💡 <strong>Tip:</strong> Click to select individual files, or drag to select multiple files at once
        </p>
      </div>

      {error && (
        <div className="bg-red-50 border border-red-200 rounded-lg p-4">
          <p className="text-red-800 font-medium">❌ {error}</p>
        </div>
      )}

      {/* File Grid */}
      {files.length === 0 ? (
        <div className="text-center py-12 bg-gray-50 rounded-lg border-2 border-dashed border-gray-300">
          <FolderOpen className="w-16 h-16 mx-auto text-gray-400 mb-4" />
          <p className="text-gray-600">No files found</p>
          <p className="text-sm text-gray-500 mt-2">
            Downloaded genomes and prediction results will appear here
          </p>
        </div>
      ) : (
        <div
          ref={containerRef}
          onMouseDown={handleMouseDown}
          className="relative grid grid-cols-2 md:grid-cols-3 lg:grid-cols-4 gap-4 select-none"
          style={{ cursor: isDragging ? 'crosshair' : 'default' }}
        >
          {files.map((file) => (
            <div
              key={file.path}
              ref={el => fileRefs.current[file.path] = el}
              onClick={(e) => {
                e.stopPropagation();
                toggleFileSelection(file.path);
              }}
              className={`file-card p-4 border-2 rounded-lg cursor-pointer transition-all ${
                selectedFiles.has(file.path)
                  ? 'border-blue-500 bg-blue-50'
                  : 'border-gray-200 hover:border-blue-300 bg-white'
              }`}
            >
              <div className="flex items-start justify-between mb-2">
                <span className="text-2xl">{getFileIcon(file.type)}</span>
                <input
                  type="checkbox"
                  checked={selectedFiles.has(file.path)}
                  onChange={() => {}}
                  className="w-5 h-5"
                  onClick={(e) => e.stopPropagation()}
                />
              </div>
              <p className="text-sm font-medium text-gray-900 truncate" title={file.filename}>
                {file.filename}
              </p>
              <div className="mt-2 space-y-1 text-xs text-gray-500">
                <p>{formatSize(file.size)}</p>
                <p>{formatDate(file.created)}</p>
                <p className="capitalize">{file.type}</p>
              </div>
            </div>
          ))}

          {/* Selection Box */}
          {isDragging && selectionBox && (
            <div
              className="absolute border-2 border-blue-500 bg-blue-100 bg-opacity-30 pointer-events-none"
              style={{
                left: Math.min(selectionBox.startX, selectionBox.endX),
                top: Math.min(selectionBox.startY, selectionBox.endY),
                width: Math.abs(selectionBox.endX - selectionBox.startX),
                height: Math.abs(selectionBox.endY - selectionBox.startY),
              }}
            />
          )}
        </div>
      )}

      {/* Delete Confirmation Modal */}
      {showConfirmDelete && (
        <div
          className="fixed inset-0 bg-black bg-opacity-50 z-50 flex items-center justify-center p-4"
          onClick={() => !deleting && setShowConfirmDelete(false)}
        >
          <div
            className="bg-white rounded-lg max-w-md w-full p-6"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center gap-3 mb-4">
              <AlertTriangle className="w-6 h-6 text-red-600" />
              <h3 className="text-lg font-bold text-gray-900">Confirm Deletion</h3>
            </div>
            
            <p className="text-gray-700 mb-6">
              {selectedFiles.size === files.length ? (
                <>Are you sure you want to delete <strong>ALL {files.length} files</strong>? This action cannot be undone.</>
              ) : (
                <>Are you sure you want to delete <strong>{selectedFiles.size} selected file(s)</strong>? This action cannot be undone.</>
              )}
            </p>

            <div className="flex gap-3">
              <button
                onClick={() => setShowConfirmDelete(false)}
                disabled={deleting}
                className="flex-1 px-4 py-2 text-gray-700 bg-gray-100 rounded-lg hover:bg-gray-200 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={selectedFiles.size === files.length ? handleCleanupAll : handleDelete}
                disabled={deleting}
                className="flex-1 px-4 py-2 text-white bg-red-600 rounded-lg hover:bg-red-700 disabled:opacity-50 flex items-center justify-center gap-2"
              >
                {deleting ? (
                  <>
                    <Loader className="w-4 h-4 animate-spin" />
                    Deleting...
                  </>
                ) : (
                  <>
                    <Trash2 className="w-4 h-4" />
                    Delete
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FileManager;