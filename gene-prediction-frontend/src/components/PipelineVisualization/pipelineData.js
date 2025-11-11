import { Database, FileText, CheckCircle } from 'lucide-react';

export const MODES = [
  { 
    id: 'catalog', 
    name: 'Browse Catalog', 
    icon: Database, 
    desc: '100 well-studied genomes' 
  },
  { 
    id: 'ncbi', 
    name: 'NCBI Download', 
    icon: Database, 
    desc: 'Any NCBI accession' 
  },
  { 
    id: 'fasta', 
    name: 'Your FASTA', 
    icon: FileText, 
    desc: 'Analyze your genome' 
  },
  { 
    id: 'validate', 
    name: 'Validate', 
    icon: CheckCircle, 
    desc: 'Compare to reference' 
  }
];

export const PIPELINE_STEPS = [
  { 
    name: 'Browse Catalog', 
    desc: 'Select genome from 100 curated organisms', 
    color: 'bg-blue-500',
    needsCatalog: true,
    details: "Interactive browser for well-studied bacterial and archaeal genomes organized by taxonomy."
  },
  { 
    name: 'Enter NCBI Accession', 
    desc: 'Provide NCBI accession number', 
    color: 'bg-blue-500',
    needsNcbi: true,
    details: "Enter an NCBI accession (e.g., NC_000913.3) to download genome directly from NCBI database."
  },
  { 
    name: 'Download from NCBI', 
    desc: 'Fetch genome sequence from NCBI', 
    color: 'bg-blue-500',
    needsCatalog: true,
    needsNcbi: true,
    details: "Downloads FASTA file from NCBI Entrez database and load it. Requires email for NCBI compliance."
  },
  { 
    name: 'Load Local FASTA', 
    desc: 'Read FASTA file from disk', 
    color: 'bg-blue-500',
    needsFasta: true,
    details: "Loads user-provided FASTA file. Supports .fasta, .fa, .fna, and .gz formats."
  },
  
  { 
    name: 'Parse FASTA', 
    desc: 'Extract DNA sequence', 
    color: 'bg-blue-500',
    alwaysShow: true,
    details: "Parses FASTA format to extract DNA sequence. Validates sequence and handles multi-line format."
  },
  { 
    name: 'Find ORFs', 
    desc: 'Identify all open reading frames (min 100bp)', 
    color: 'bg-blue-500',
    alwaysShow: true,
    details: "Scans both strands for ATG/GTG/TTG start codons and TAA/TAG/TGA stop codons. Minimum ORF length: 100bp."
  },
  { 
    name: 'Training Set', 
    desc: 'Create training set from long ORFs', 
    color: 'bg-purple-500',
    alwaysShow: true,
    details: "Selects long ORFs (>300bp by default) as likely real genes to train statistical models."
  },
  { 
    name: 'Intergenic', 
    desc: 'Extract non-coding regions', 
    color: 'bg-purple-500',
    alwaysShow: true,
    details: "Extracts sequences between predicted genes to model non-coding DNA characteristics."
  },
  { 
    name: 'Build Models', 
    desc: 'Codon usage, RBS, start context', 
    color: 'bg-purple-500',
    alwaysShow: true,
    details: "Builds three models: (1) Codon usage frequencies, (2) RBS motif patterns, (3) Start codon context."
  },
  { 
    name: 'Score ORFs', 
    desc: 'Apply statistical models', 
    color: 'bg-green-500',
    alwaysShow: true,
    details: "Applies all scoring models to every ORF candidate. Combines codon, RBS, and context scores."
  },
  { 
    name: 'Initial Filter', 
    desc: 'Remove low-scoring candidates', 
    color: 'bg-green-500',
    alwaysShow: true,
    details: "Removes ORFs below threshold scores. Uses FIRST_FILTER_THRESHOLD from config."
  },
  { 
    name: 'Group Nested', 
    desc: 'Organize overlapping ORFs', 
    color: 'bg-yellow-500',
    alwaysShow: true,
    details: "Groups overlapping ORFs sharing the same stop codon. Handles nested gene structures."
  },
  { 
    name: 'ML Group Filter', 
    desc: 'LightGBM classification (optional)', 
    color: 'bg-red-500', 
    ml: true,
    alwaysShow: true, 
    details: "ML model classifies entire ORF groups as real/false. Uses LightGBM trained on annotated genomes.",
    note: "Can be disabled with --no-group-ml"
  },
  { 
    name: 'Select Starts', 
    desc: 'Choose best start codon per group', 
    color: 'bg-yellow-500',
    alwaysShow: true,
    details: "For each group, selects the start codon with highest combined score using weighted criteria."
  },
  { 
    name: 'Final Filter', 
    desc: 'Apply strict thresholds', 
    color: 'bg-green-500',
    alwaysShow: true,
    details: "Applies stricter thresholds (SECOND_FILTER_THRESHOLD) to final candidates."
  },
  { 
    name: 'Hybrid ML', 
    desc: 'Neural network filtration (optional)', 
    color: 'bg-red-500', 
    ml: true,
    alwaysShow: true,  
    details: "Hybrid neural network provides final quality check. Can remove remaining false positives.",
    note: "Can be disabled with --no-final-ml"
  },
  
  { 
    name: 'Write GFF', 
    desc: 'Output predictions', 
    color: 'bg-blue-500',
    alwaysShow: true,
    details: "Outputs predictions in GFF3 format with gene coordinates, scores, and metadata."
  },
  
  // ==================== VALIDATION STEPS (Validate Mode Only) ====================
  { 
    name: 'Load Prediction GFF', 
    desc: 'Read prediction results', 
    color: 'bg-orange-500',
    needsValidate: true,
    details: "Loads the GFF file containing predicted genes from previous run."
  },
  { 
    name: 'Download Reference GFF', 
    desc: 'Fetch reference annotations from NCBI', 
    color: 'bg-orange-500',
    needsValidate: true,
    details: "Downloads official gene annotations from NCBI to use as ground truth."
  },
  { 
    name: 'Compare Predictions', 
    desc: 'Match predictions to reference', 
    color: 'bg-orange-500',
    needsValidate: true,
    details: "Compares predicted genes to reference annotations. Identifies true positives, false positives, and false negatives."
  },
  { 
    name: 'Calculate Metrics', 
    desc: 'Compute precision, recall, F1', 
    color: 'bg-orange-500',
    needsValidate: true,
    details: "Calculates validation metrics: Sensitivity (Recall), Precision, and F1 Score."
  },
  { 
    name: 'Generate Report', 
    desc: 'Create validation report', 
    color: 'bg-orange-500',
    needsValidate: true,
    details: "Generates a detailed validation report saved as TXT file in results directory."
  }
];

export const ML_OPTIONS = [
  { 
    name: 'Group ML Filtering', 
    param: '--group-threshold', 
    default: '0.1', 
    desc: 'Filters ORF groups using LightGBM classifier' 
  },
  { 
    name: 'Final ML Filtration', 
    param: '--final-threshold', 
    default: '0.12', 
    desc: 'Final filtering with hybrid neural network' 
  }
];

export const FEATURES = {
  traditional: {
    title: 'Traditional Methods',
    items: [
      'Codon usage bias',
      'RBS motif detection',
      'Start codon context',
      'Statistical scoring'
    ],
    bgColor: 'bg-blue-50',
    textColor: 'text-blue-900',
    itemColor: 'text-blue-800'
  },
  ml: {
    title: 'Machine Learning',
    items: [
      'LightGBM classifier',
      'Hybrid neural network',
      'Configurable thresholds',
      'Optional filtering'
    ],
    bgColor: 'bg-red-50',
    textColor: 'text-red-900',
    itemColor: 'text-red-800'
  },
  validation: {
    title: 'Validation',
    items: [
      'Compare to reference',
      'Precision & recall',
      'F1 score metrics',
      'Automated reports'
    ],
    bgColor: 'bg-green-50',
    textColor: 'text-green-900',
    itemColor: 'text-green-800'
  }
};