import { useState, useRef, useEffect } from 'react';
import { ZoomIn, ZoomOut, Maximize2, X } from 'lucide-react';

const GenomeViewer = ({ predictions, genomeLength, genomeId, highlightGeneId, onGeneClick }) => {
  const canvasRef = useRef(null);
  const [zoom, setZoom] = useState(1);
  const [offset, setOffset] = useState(0);
  const [hoveredGene, setHoveredGene] = useState(null);
  const [selectedGene, setSelectedGene] = useState(null);
  const [selectedStrand, setSelectedStrand] = useState('both');
  const [isDragging, setIsDragging] = useState(false);
  const [dragStart, setDragStart] = useState(0);

  const canvasWidth = 1000;
  const canvasHeight = 200;
  const trackHeight = 30;
  const padding = 40;

  useEffect(() => {
    drawGenome();
  }, [predictions, zoom, offset, selectedStrand, highlightGeneId, selectedGene]);

  // Jump to highlighted gene and show its info
  useEffect(() => {
    if (highlightGeneId) {
      const gene = predictions.find(g => g.gene_id === highlightGeneId);
      if (gene) {
        setSelectedGene(highlightGeneId);
        
        const geneCenter = (gene.start + gene.end) / 2;
        const visibleLength = genomeLength / zoom;
        const newOffset = Math.max(0, Math.min(geneCenter - visibleLength / 2, genomeLength - visibleLength));
        setOffset(newOffset);
      }
    }
  }, [highlightGeneId]);

  // Keyboard controls
  useEffect(() => {
    const handleKeyDown = (e) => {
      const scrollAmount = (genomeLength / zoom) * 0.05;
      const maxOffset = genomeLength - (genomeLength / zoom);
      
      if (e.key === 'ArrowLeft') {
        e.preventDefault();
        setOffset(Math.max(0, offset - scrollAmount));
      } else if (e.key === 'ArrowRight') {
        e.preventDefault();
        setOffset(Math.min(maxOffset, offset + scrollAmount));
      }
    };

    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [offset, zoom, genomeLength]);

  const drawGenome = () => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, canvasWidth, canvasHeight);

    const visibleLength = genomeLength / zoom;
    const startPos = offset;
    const endPos = startPos + visibleLength;

    // Draw background
    ctx.fillStyle = '#f9fafb';
    ctx.fillRect(0, 0, canvasWidth, canvasHeight);

    // Draw both strand tracks (ALWAYS visible)
    ctx.strokeStyle = '#d1d5db';
    ctx.lineWidth = 2;
    
    ctx.beginPath();
    ctx.moveTo(padding, canvasHeight / 2 - trackHeight - 5);
    ctx.lineTo(canvasWidth - padding, canvasHeight / 2 - trackHeight - 5);
    ctx.stroke();
    
    ctx.beginPath();
    ctx.moveTo(padding, canvasHeight / 2 + trackHeight + 5);
    ctx.lineTo(canvasWidth - padding, canvasHeight / 2 + trackHeight + 5);
    ctx.stroke();

    // Draw genome axis (center)
    ctx.strokeStyle = '#9ca3af';
    ctx.lineWidth = 2;
    ctx.beginPath();
    ctx.moveTo(padding, canvasHeight / 2);
    ctx.lineTo(canvasWidth - padding, canvasHeight / 2);
    ctx.stroke();

    // Draw scale markers
    const numMarkers = 10;
    const markerInterval = visibleLength / numMarkers;
    ctx.fillStyle = '#6b7280';
    ctx.font = '10px monospace';

    for (let i = 0; i <= numMarkers; i++) {
      const pos = startPos + (i * markerInterval);
      const x = padding + (i / numMarkers) * (canvasWidth - 2 * padding);
      
      ctx.strokeStyle = '#9ca3af';
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(x, canvasHeight / 2 - 5);
      ctx.lineTo(x, canvasHeight / 2 + 5);
      ctx.stroke();

      ctx.fillText(
        `${Math.round(pos / 1000)}kb`,
        x - 15,
        canvasHeight / 2 + 20
      );
    }

    // Filter genes by selected strand
    const visibleGenes = predictions.filter(gene => {
      if (selectedStrand === 'forward' && gene.strand !== 'forward') return false;
      if (selectedStrand === 'reverse' && gene.strand !== 'reverse') return false;
      
      return gene.end >= startPos && gene.start <= endPos;
    });

    // Draw genes
    visibleGenes.forEach(gene => {
      const geneStart = Math.max(gene.start, startPos);
      const geneEnd = Math.min(gene.end, endPos);
      
      const x1 = padding + ((geneStart - startPos) / visibleLength) * (canvasWidth - 2 * padding);
      const x2 = padding + ((geneEnd - startPos) / visibleLength) * (canvasWidth - 2 * padding);
      
      const width = Math.max(x2 - x1, 2);

      const y = gene.strand === 'forward' 
        ? canvasHeight / 2 - trackHeight - 5 - trackHeight / 2
        : canvasHeight / 2 + trackHeight + 5 + trackHeight / 2;

      const isSelected = gene.gene_id === selectedGene;
      const isHovered = hoveredGene === gene.gene_id;

      let fillColor;
      if (isSelected) {
        fillColor = '#f59e0b';
      } else if (isHovered) {
        fillColor = '#3b82f6';
      } else {
        fillColor = getColorForScore(gene.combined_score);
      }
      
      ctx.fillStyle = fillColor;
      
      const geneHeight = Math.min(trackHeight, 20);
      ctx.fillRect(x1, y - geneHeight / 2, width, geneHeight);
      
      if (width > 10) {
        ctx.fillStyle = '#ffffff';
        const arrowSize = Math.min(5, width / 3);
        ctx.beginPath();
        if (gene.strand === 'forward') {
          ctx.moveTo(x2 - arrowSize, y - geneHeight / 2 + 3);
          ctx.lineTo(x2, y);
          ctx.lineTo(x2 - arrowSize, y + geneHeight / 2 - 3);
        } else {
          ctx.moveTo(x1 + arrowSize, y - geneHeight / 2 + 3);
          ctx.lineTo(x1, y);
          ctx.lineTo(x1 + arrowSize, y + geneHeight / 2 - 3);
        }
        ctx.fill();
      }

      ctx.strokeStyle = isSelected ? '#f59e0b' : '#1f2937';
      ctx.lineWidth = isSelected ? 3 : 1;
      ctx.strokeRect(x1, y - geneHeight / 2, width, geneHeight);
    });

    // Draw legend - positioned in corners
    ctx.fillStyle = '#374151';
    ctx.font = 'bold 11px sans-serif';
    ctx.fillText('Forward →', 10, 15);
    ctx.fillText('← Reverse', 10, canvasHeight - 10);
  };

  const getColorForScore = (score) => {
    if (score > 0.8) return '#10b981';
    if (score > 0.6) return '#84cc16';
    if (score > 0.4) return '#eab308';
    if (score > 0.2) return '#f97316';
    return '#ef4444';
  };

  const handleMouseMove = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    const scaleX = canvas.width / rect.width;
    const x = (e.clientX - rect.left) * scaleX;

    const visibleLength = genomeLength / zoom;
    const genomePos = offset + ((x - padding) / (canvasWidth - 2 * padding)) * visibleLength;

    const gene = predictions.find(g => {
      if (selectedStrand === 'forward' && g.strand !== 'forward') return false;
      if (selectedStrand === 'reverse' && g.strand !== 'reverse') return false;
      return g.start <= genomePos && g.end >= genomePos;
    });

    setHoveredGene(gene ? gene.gene_id : null);
  };

  const handleMouseLeave = () => {
    setHoveredGene(null);
  };

  const handleMouseDown = (e) => {
    setIsDragging(true);
    setDragStart(e.clientX);
  };

  const handleMouseMoveCanvas = (e) => {
    if (isDragging) {
      const canvas = canvasRef.current;
      const rect = canvas.getBoundingClientRect();
      const scaleX = canvas.width / rect.width;
      
      const deltaX = (e.clientX - dragStart) * scaleX;
      const visibleLength = genomeLength / zoom;
      const genomeDelta = -(deltaX / (canvasWidth - 2 * padding)) * visibleLength;
      
      const maxOffset = genomeLength - visibleLength;
      setOffset(Math.max(0, Math.min(offset + genomeDelta, maxOffset)));
      setDragStart(e.clientX);
    } else {
      handleMouseMove(e);
    }
  };

  const handleMouseUp = () => {
    setIsDragging(false);
  };

  const handleClick = (e) => {
    const canvas = canvasRef.current;
    const rect = canvas.getBoundingClientRect();
    
    const scaleX = canvas.width / rect.width;
    const x = (e.clientX - rect.left) * scaleX;

    const visibleLength = genomeLength / zoom;
    const genomePos = offset + ((x - padding) / (canvasWidth - 2 * padding)) * visibleLength;

    const clickedGene = predictions.find(g => {
      if (selectedStrand === 'forward' && g.strand !== 'forward') return false;
      if (selectedStrand === 'reverse' && g.strand !== 'reverse') return false;
      return g.start <= genomePos && g.end >= genomePos;
    });

    if (clickedGene) {
      const geneId = clickedGene.gene_id;
      if (selectedGene === geneId) {
        setSelectedGene(null);
        if (onGeneClick) {
          onGeneClick(null);
        }
      } else {
        setSelectedGene(geneId);
        if (onGeneClick) {
          onGeneClick(geneId);
        }
      }
    }
  };

  const handleZoomIn = () => {
    setZoom(Math.min(zoom * 2, 100));
  };

  const handleZoomOut = () => {
    setZoom(Math.max(zoom / 2, 1));
  };

  const handleReset = () => {
    setZoom(1);
    setOffset(0);
  };

  const handleScroll = (e) => {
    e.preventDefault(); 
    e.stopPropagation(); 
  
    const scrollAmount = e.deltaY * (genomeLength / zoom) * 0.001;
    const maxOffset = genomeLength - (genomeLength / zoom);
    setOffset(Math.max(0, Math.min(offset + scrollAmount, maxOffset)));
  };

  const handleDeselect = () => {
    setSelectedGene(null);
    if (onGeneClick) {
      onGeneClick(null);
    }
  };

  // Calculate genome stats
  const forwardGenes = predictions.filter(g => g.strand === 'forward').length;
  const reverseGenes = predictions.filter(g => g.strand === 'reverse').length;

  // Determine what to show in info panel
  const displayGeneId = selectedGene || hoveredGene;
  const displayGeneData = displayGeneId 
    ? predictions.find(g => g.gene_id === displayGeneId)
    : null;

  return (
    <div className="space-y-4">
      {/* Controls */}
      <div className="flex items-center justify-between bg-gray-50 rounded-lg p-4">
        <div className="flex items-center gap-4">
          <span className="text-sm font-medium text-gray-700">Strand:</span>
          <select
            value={selectedStrand}
            onChange={(e) => setSelectedStrand(e.target.value)}
            className="px-3 py-1 border border-gray-300 rounded-lg text-sm"
          >
            <option value="both">Both Strands</option>
            <option value="forward">Forward Only</option>
            <option value="reverse">Reverse Only</option>
          </select>

          <div className="ml-4 flex items-center gap-2">
            <button
              onClick={handleZoomOut}
              disabled={zoom <= 1}
              className="p-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ZoomOut className="w-4 h-4" />
            </button>
            <span className="text-sm text-gray-600 min-w-[60px] text-center">
              {zoom.toFixed(1)}x
            </span>
            <button
              onClick={handleZoomIn}
              disabled={zoom >= 100}
              className="p-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-100 disabled:opacity-50 disabled:cursor-not-allowed"
            >
              <ZoomIn className="w-4 h-4" />
            </button>
            <button
              onClick={handleReset}
              className="p-2 text-gray-700 bg-white border border-gray-300 rounded-lg hover:bg-gray-100"
            >
              <Maximize2 className="w-4 h-4" />
            </button>
          </div>
        </div>

        <div className="text-sm text-gray-600">
          <span className="font-medium">{genomeId}</span>
          <span className="ml-3">{genomeLength.toLocaleString()} bp</span>
          <span className="ml-3">{predictions.length} genes</span>
        </div>
      </div>

      {/* Canvas */}
      <div className="bg-white rounded-lg border border-gray-200 p-4">
        <canvas
          ref={canvasRef}
          width={canvasWidth}
          height={canvasHeight}
          onMouseMove={handleMouseMoveCanvas}
          onMouseDown={handleMouseDown}
          onMouseUp={handleMouseUp}
          onMouseLeave={() => {
            handleMouseLeave();
            setIsDragging(false);
          }}
          onClick={handleClick}
          onWheel={handleScroll}
          className={isDragging ? 'cursor-grabbing' : 'cursor-grab'}
          style={{ width: '100%', height: 'auto' }}
          tabIndex={0}
        />
        
        {/* PERSISTENT Info Panel - FIXED HEIGHT */}
        <div className={`mt-4 p-4 rounded-lg transition-colors h-[250px] flex flex-col ${
          selectedGene 
            ? 'bg-amber-50 border-2 border-amber-400' 
            : hoveredGene
            ? 'bg-blue-50 border-2 border-blue-300'
            : 'bg-gray-50 border-2 border-gray-300'
        }`}>
          <div className="flex items-start justify-between mb-3 flex-shrink-0">
            <h4 className={`font-semibold ${
              selectedGene 
                ? 'text-amber-900' 
                : hoveredGene
                ? 'text-blue-900'
                : 'text-gray-900'
            }`}>
              {selectedGene 
                ? '📌 Selected Gene' 
                : hoveredGene
                ? 'Gene Information (hover)'
                : '📊 Genome Statistics'}
            </h4>
            {selectedGene && (
              <button 
                onClick={handleDeselect}
                className="text-amber-600 hover:text-amber-800 flex-shrink-0"
              >
                <X className="w-4 h-4" />
              </button>
            )}
          </div>

          <div className="flex-1 min-h-0">
            {displayGeneData ? (
              <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                <span className="text-gray-600">Gene:</span>
                <span className="font-semibold text-gray-900">{displayGeneData.gene_id}</span>
                
                <span className="text-gray-600">Position:</span>
                <span className="font-mono text-gray-900">
                  {displayGeneData.start.toLocaleString()} - {displayGeneData.end.toLocaleString()}
                </span>
                
                <span className="text-gray-600">Length:</span>
                <span className="text-gray-900">{displayGeneData.length.toLocaleString()} bp</span>
                
                <span className="text-gray-600">Strand:</span>
                <span className="text-gray-900">
                  {displayGeneData.strand === 'forward' ? '→ Forward' : '← Reverse'}
                </span>
                
                <span className="text-gray-600">Score:</span>
                <span className="text-gray-900">{displayGeneData.combined_score.toFixed(3)}</span>
                
                <span className="text-gray-600">RBS Score:</span>
                <span className="text-gray-900">
                  {displayGeneData.rbs_score ? displayGeneData.rbs_score.toFixed(2) : 'N/A'}
                </span>
              </div>
            ) : (
              <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm">
                <span className="text-gray-600">Total Genes:</span>
                <span className="font-semibold text-gray-900">{predictions.length}</span>
                
                <span className="text-gray-600">Forward Strand:</span>
                <span className="text-blue-600 font-semibold">→ {forwardGenes} genes</span>
                
                <span className="text-gray-600">Reverse Strand:</span>
                <span className="text-purple-600 font-semibold">← {reverseGenes} genes</span>
                
                <span className="text-gray-600">Genome Length:</span>
                <span className="text-gray-900">{genomeLength.toLocaleString()} bp</span>
                
                <span className="text-gray-600">Gene Density:</span>
                <span className="text-gray-900">
                  {(predictions.length / (genomeLength / 1000)).toFixed(2)} genes/kb
                </span>
                
                <span className="text-gray-600">&nbsp;</span>
                <span className="text-gray-900">&nbsp;</span>
              </div>
            )}
          </div>

          <div className="mt-4 flex-shrink-0">
            {!selectedGene && hoveredGene && (
              <p className="text-xs text-blue-600">💡 Click to pin this gene</p>
            )}
            {selectedGene && (
              <p className="text-xs text-amber-600">💡 Click X or click gene again to deselect</p>
            )}
            {!selectedGene && !hoveredGene && (
              <p className="text-xs text-gray-500">💡 Hover over genes to view details</p>
            )}
          </div>
        </div>

        {/* Color Legend */}
        <div className="mt-4 flex items-center gap-4 text-xs text-gray-600">
          <span className="font-medium">Score:</span>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-red-500 rounded"></div>
            <span>Low (0-0.2)</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-orange-500 rounded"></div>
            <span>0.2-0.4</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-yellow-500 rounded"></div>
            <span>0.4-0.6</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-lime-500 rounded"></div>
            <span>0.6-0.8</span>
          </div>
          <div className="flex items-center gap-2">
            <div className="w-4 h-4 bg-green-500 rounded"></div>
            <span>High (0.8+)</span>
          </div>
          <div className="flex items-center gap-2 ml-4">
            <div className="w-4 h-4 bg-amber-500 rounded border-2 border-amber-600"></div>
            <span>Selected</span>
          </div>
        </div>

        <p className="mt-2 text-xs text-gray-500">
          💡 Arrow keys or drag to pan • Mouse wheel to scroll • Click to select
        </p>
      </div>
    </div>
  );
};

export default GenomeViewer;