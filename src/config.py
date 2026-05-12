"""
Configuration Module for Bacterial Gene Prediction

Global constants and configuration settings.
Centralized location for all project-wide constants.
"""

__all__ = [
    "NCBI_EMAIL",
    "TEST_GENOMES",
    "GENOME_CATALOG",
    "START_CODON_WEIGHTS",
    "START_SELECTION_WEIGHTS",
    "FIRST_FILTER_THRESHOLD",
    "SECOND_FILTER_THRESHOLD",
    "SCORE_WEIGHTS",
    "START_CODONS",
    "STOP_CODONS",
    "get_genome_by_id",
    "get_genome_by_accession",
    "list_genomes_by_group",
]

# NCBI Configuration
NCBI_EMAIL = "your_email@example.com"

# Holdout benchmark genomes — completely disjoint from GENOME_CATALOG.
# These 20 genomes were NEVER used in any training or validation run.
# 5 genomes per taxonomic group, chosen from distinct genera to maximise
# diversity while avoiding overlap with the training pool.
# Use ONLY this list for benchmark evaluation — never add GENOME_CATALOG
# entries here and never add these entries to GENOME_CATALOG.
TEST_GENOMES = [
    # ── Proteobacteria (5) — 5 distinct genera, no overlap with GENOME_CATALOG
    "NC_002947.4",  # Pseudomonas putida KT2440             (Gammaproteobacteria)
    "NC_002929.2",  # Bordetella pertussis Tohama I          (Betaproteobacteria)
    "NC_003143.1",  # Yersinia pestis CO92                   (Gammaproteobacteria)
    "NC_003116.1",  # Neisseria meningitidis Z2491           (Betaproteobacteria)
    "NC_004757.1",  # Nitrosomonas europaea ATCC 19718       (Betaproteobacteria)
    # ── Firmicutes (5) — distinct genera (Lactobacillus/Streptococcus/Bacillus/Clostridium)
    "NC_008497.1",  # Lactobacillus brevis ATCC 367
    "NC_004350.2",  # Streptococcus agalactiae A909
    "NC_006270.3",  # Bacillus licheniformis DSM 13
    "NC_006274.1",  # Bacillus cereus E33L
    "NC_003030.1",  # Clostridium acetobutylicum ATCC 824
    # ── Actinobacteria (5) — 5 distinct genera, no overlap with GENOME_CATALOG
    "NC_003155.5",  # Streptomyces avermitilis MA-4680
    "NC_003450.3",  # Corynebacterium glutamicum ATCC 13032
    "NC_002677.1",  # Mycobacterium leprae TN
    "NC_008268.1",  # Nocardia farcinica IFM 10152
    "NC_006958.1",  # Corynebacterium glutamicum R
    # ── Archaea (5) — Hyperthermus/Haloarcula/Methanobrevibacter/Methanosaeta/Methanoculleus
    "NC_008818.1",  # Hyperthermus butylicus DSM 5456        (Crenarchaeota)
    "NC_015948.1",  # Haloarcula hispanica ATCC 33960        (Euryarchaeota)
    "NC_014408.1",  # Methanobrevibacter ruminantium M1      (Euryarchaeota)
    "NC_019977.1",  # Methanosaeta harundinacea 6Ac          (Euryarchaeota)
    "NC_007644.1",  # Methanoculleus marisnigri JR1          (Euryarchaeota)
]

LENGTH_REFERENCE_BP = 300  # Reference length for scaling
MIN_ORF_LENGTH = 100  # Minimum length to prevent log(0) errors

# RBS Detection Parameters
RBS_UPSTREAM_LENGTH = 20  # How far upstream to search for RBS
RBS_MIN_PURINE_CONTENT = 0.6  # Minimum purine content for RBS

# Known RBS (Shine-Dalgarno) motifs
KNOWN_RBS_MOTIFS = ["AGGAGG", "GGAGG", "AGGAG", "GAGG", "AGGA", "GGAG"]

# Start and stop codons
START_CODONS = {"ATG", "GTG", "TTG"}
STOP_CODONS = {"TAA", "TAG", "TGA"}

# Cache Settings
CACHE_FILENAME = "cached_orfs.pkl"

# Equal weights are intentional: each component is z-score normalised before
# this step, so all dimensions are already on the same scale and no additional
# biological weighting is required here.  Empirically calibrated weights for
# the later start-site selection step live in START_SELECTION_WEIGHTS below.
SCORE_WEIGHTS = {
    "codon": 1.0,
    "imm": 1.0,
    "rbs": 1.0,
    "length": 1.0,
    "start": 1.0,
}


START_CODON_WEIGHTS = {
    "ATG": 1.0,
    "GTG": 0.7,
    "TTG": 0.4,
}

START_SELECTION_WEIGHTS = {
    "codon": 4.8562,
    "imm": 1.0107,
    "rbs": 0.6383,
    "length": 7.4367,
    "start": 0.2755,
}

FIRST_FILTER_THRESHOLD = {
    "codon_threshold": 0.2665,
    "imm_threshold": 0.2308,
    "length_threshold": -0.2812,
    "combined_threshold": 0.2643,
}

SECOND_FILTER_THRESHOLD = {
    "codon_threshold": 0.0872,
    "imm_threshold": 0.3157,
    "length_threshold": 0.2762,
    "combined_threshold": 0.4669,
}

"""
Curated Genome Catalog for Easy Testing
100 well-studied bacteria and archaea from major taxonomic groups
"""

GENOME_CATALOG = [
    # Proteobacteria (25 genomes) - Largest bacterial phylum
    {
        "id": 1,
        "accession": "NC_000913.3",
        "name": "Escherichia coli K-12 MG1655",
        "group": "Proteobacteria",
    },
    {
        "id": 2,
        "accession": "NC_002695.2",
        "name": "Escherichia coli O157:H7 Sakai",
        "group": "Proteobacteria",
    },
    {
        "id": 3,
        "accession": "NC_008253.1",
        "name": "Escherichia coli 536",
        "group": "Proteobacteria",
    },
    {
        "id": 4,
        "accession": "NC_003197.2",
        "name": "Salmonella enterica LT2",
        "group": "Proteobacteria",
    },
    {
        "id": 5,
        "accession": "NC_002505.1",
        "name": "Vibrio cholerae",
        "group": "Proteobacteria",
    },
    {
        "id": 6,
        "accession": "NC_002516.2",
        "name": "Pseudomonas aeruginosa PAO1",
        "group": "Proteobacteria",
    },
    {
        "id": 7,
        "accession": "NC_004741.1",
        "name": "Shigella flexneri 2a",
        "group": "Proteobacteria",
    },
    {
        "id": 8,
        "accession": "NC_007503.1",
        "name": "Carboxydothermus hydrogenoformans",
        "group": "Proteobacteria",
    },
    {
        "id": 9,
        "accession": "NC_008570.1",
        "name": "Aeromonas hydrophila",
        "group": "Proteobacteria",
    },
    {
        "id": 10,
        "accession": "NC_010067.1",
        "name": "Salmonella Typhimurium",
        "group": "Proteobacteria",
    },
    {
        "id": 11,
        "accession": "NC_010473.1",
        "name": "Escherichia coli K-12 DH10B",
        "group": "Proteobacteria",
    },
    {
        "id": 12,
        "accession": "NC_011750.1",
        "name": "Escherichia coli IAI39",
        "group": "Proteobacteria",
    },
    {
        "id": 13,
        "accession": "NC_012759.1",
        "name": "Escherichia coli BW2952",
        "group": "Proteobacteria",
    },
    {
        "id": 14,
        "accession": "NC_013654.1",
        "name": "Shigella boydii",
        "group": "Proteobacteria",
    },
    {
        "id": 15,
        "accession": "NC_004547.2",
        "name": "Erwinia carotovora",
        "group": "Proteobacteria",
    },
    {
        "id": 16,
        "accession": "NC_007946.1",
        "name": "Escherichia coli UTI89",
        "group": "Proteobacteria",
    },
    {
        "id": 17,
        "accession": "NC_009648.1",
        "name": "Klebsiella pneumoniae 342",
        "group": "Proteobacteria",
    },
    {
        "id": 18,
        "accession": "NC_010498.1",
        "name": "Escherichia coli SMS-3-5",
        "group": "Proteobacteria",
    },
    {
        "id": 19,
        "accession": "NC_011294.1",
        "name": "Salmonella enterica P125109",
        "group": "Proteobacteria",
    },
    {
        "id": 20,
        "accession": "NC_011415.1",
        "name": "Escherichia coli SE11",
        "group": "Proteobacteria",
    },
    {
        "id": 21,
        "accession": "NC_012967.1",
        "name": "Escherichia coli B str. REL606",
        "group": "Proteobacteria",
    },
    {
        "id": 22,
        "accession": "NC_013361.1",
        "name": "Escherichia coli O26:H11",
        "group": "Proteobacteria",
    },
    {
        "id": 23,
        "accession": "NC_013941.1",
        "name": "Escherichia coli O55:H7",
        "group": "Proteobacteria",
    },
    {
        "id": 24,
        "accession": "NC_017634.1",
        "name": "Salmonella enterica RKS4594",
        "group": "Proteobacteria",
    },
    {
        "id": 25,
        "accession": "NC_020063.1",
        "name": "Escherichia coli O145:H28",
        "group": "Proteobacteria",
    },
    # Firmicutes (25 genomes) - Gram-positive bacteria
    {
        "id": 26,
        "accession": "NC_000964.3",
        "name": "Bacillus subtilis 168",
        "group": "Firmicutes",
    },
    {
        "id": 27,
        "accession": "NC_003210.1",
        "name": "Listeria monocytogenes EGD-e",
        "group": "Firmicutes",
    },
    {
        "id": 28,
        "accession": "NC_002737.2",
        "name": "Streptococcus pyogenes M1 GAS",
        "group": "Firmicutes",
    },
    {
        "id": 29,
        "accession": "NC_003098.1",
        "name": "Streptococcus pneumoniae R6",
        "group": "Firmicutes",
    },
    {
        "id": 30,
        "accession": "NC_004461.1",
        "name": "Staphylococcus epidermidis",
        "group": "Firmicutes",
    },
    {
        "id": 31,
        "accession": "NC_007168.1",
        "name": "Staphylococcus haemolyticus",
        "group": "Firmicutes",
    },
    {
        "id": 32,
        "accession": "NC_007795.1",
        "name": "Staphylococcus aureus NCTC 8325",
        "group": "Firmicutes",
    },
    {
        "id": 33,
        "accession": "NC_009641.1",
        "name": "Staphylococcus aureus Newman",
        "group": "Firmicutes",
    },
    {
        "id": 34,
        "accession": "NC_010079.1",
        "name": "Staphylococcus aureus TCH1516",
        "group": "Firmicutes",
    },
    {
        "id": 35,
        "accession": "NC_013891.1",
        "name": "Listeria seeligeri",
        "group": "Firmicutes",
    },
    {
        "id": 36,
        "accession": "NC_014829.1",
        "name": "Bacillus cellulosilyticus",
        "group": "Firmicutes",
    },
    {
        "id": 37,
        "accession": "NC_016047.1",
        "name": "Bacillus subtilis TU-B-10",
        "group": "Firmicutes",
    },
    {
        "id": 38,
        "accession": "NC_017195.1",
        "name": "Bacillus subtilis TO-A",
        "group": "Firmicutes",
    },
    {
        "id": 39,
        "accession": "NC_018520.1",
        "name": "Bacillus subtilis XF-1",
        "group": "Firmicutes",
    },
    {
        "id": 40,
        "accession": "NC_020244.1",
        "name": "Bacillus subtilis QB928",
        "group": "Firmicutes",
    },
    {
        "id": 41,
        "accession": "NC_004668.1",
        "name": "Enterococcus faecalis V583",
        "group": "Firmicutes",
    },
    {
        "id": 42,
        "accession": "NC_008555.1",
        "name": "Listeria welshimeri",
        "group": "Firmicutes",
    },
    {
        "id": 43,
        "accession": "NC_012469.1",
        "name": "Streptococcus mutans NN2025",
        "group": "Firmicutes",
    },
    {
        "id": 44,
        "accession": "NC_013890.1",
        "name": "Streptococcus suis",
        "group": "Firmicutes",
    },
    {
        "id": 45,
        "accession": "NC_014498.1",
        "name": "Streptococcus pneumoniae 670",
        "group": "Firmicutes",
    },
    {
        "id": 46,
        "accession": "NC_015875.1",
        "name": "Streptococcus pseudopneumoniae",
        "group": "Firmicutes",
    },
    {
        "id": 47,
        "accession": "NC_017349.1",
        "name": "Staphylococcus aureus LGA251",
        "group": "Firmicutes",
    },
    {
        "id": 48,
        "accession": "NC_017673.1",
        "name": "Staphylococcus aureus 71193",
        "group": "Firmicutes",
    },
    {
        "id": 49,
        "accession": "NC_018608.1",
        "name": "Staphylococcus aureus 08BA02176",
        "group": "Firmicutes",
    },
    {
        "id": 50,
        "accession": "NC_020291.1",
        "name": "Clostridium difficile",
        "group": "Firmicutes",
    },
    # Actinobacteria (25 genomes) - High GC Gram-positives
    {
        "id": 51,
        "accession": "NC_000962.3",
        "name": "Mycobacterium tuberculosis H37Rv",
        "group": "Actinobacteria",
    },
    {
        "id": 52,
        "accession": "NC_002755.2",
        "name": "Mycobacterium tuberculosis CDC1551",
        "group": "Actinobacteria",
    },
    {
        "id": 53,
        "accession": "NC_008769.1",
        "name": "Mycobacterium bovis BCG Pasteur",
        "group": "Actinobacteria",
    },
    {
        "id": 54,
        "accession": "NC_009525.1",
        "name": "Mycobacterium tuberculosis H37Ra",
        "group": "Actinobacteria",
    },
    {
        "id": 55,
        "accession": "NC_012207.1",
        "name": "Mycobacterium bovis AF2122/97",
        "group": "Actinobacteria",
    },
    {
        "id": 56,
        "accession": "NC_015848.1",
        "name": "Mycobacterium canettii",
        "group": "Actinobacteria",
    },
    {
        "id": 57,
        "accession": "NC_016804.1",
        "name": "Mycobacterium bovis BCG Tokyo",
        "group": "Actinobacteria",
    },
    {
        "id": 58,
        "accession": "NC_017026.1",
        "name": "Mycobacterium tuberculosis RGTB327",
        "group": "Actinobacteria",
    },
    {
        "id": 59,
        "accession": "NC_018143.2",
        "name": "Mycobacterium tuberculosis KZN 1435",
        "group": "Actinobacteria",
    },
    {
        "id": 60,
        "accession": "NC_019950.1",
        "name": "Mycobacterium canettii CIPT",
        "group": "Actinobacteria",
    },
    {
        "id": 61,
        "accession": "NC_003888.3",
        "name": "Streptomyces coelicolor A3(2)",
        "group": "Actinobacteria",
    },
    {
        "id": 62,
        "accession": "NC_009664.1",
        "name": "Kineococcus radiotolerans",
        "group": "Actinobacteria",
    },
    {
        "id": 63,
        "accession": "NC_013131.1",
        "name": "Catenulispora acidiphila",
        "group": "Actinobacteria",
    },
    {
        "id": 64,
        "accession": "NC_013235.1",
        "name": "Nakamurella multipartita",
        "group": "Actinobacteria",
    },
    {
        "id": 65,
        "accession": "NC_013235.1",
        "name": "Nakamurella multipartita",
        "group": "Actinobacteria",
    },
    {
        "id": 66,
        "accession": "NC_014158.1",
        "name": "Tsukamurella paurometabola",
        "group": "Actinobacteria",
    },
    {
        "id": 67,
        "accession": "NC_014666.1",
        "name": "Frankia sp. EAN1pec",
        "group": "Actinobacteria",
    },
    {
        "id": 68,
        "accession": "NC_015312.1",
        "name": "Pseudonocardia dioxanivorans",
        "group": "Actinobacteria",
    },
    {
        "id": 69,
        "accession": "NC_016111.1",
        "name": "Streptomyces cattleya",
        "group": "Actinobacteria",
    },
    {
        "id": 70,
        "accession": "NC_016582.1",
        "name": "Streptomyces bingchenggensis",
        "group": "Actinobacteria",
    },
    {
        "id": 71,
        "accession": "NC_017093.1",
        "name": "Actinoplanes missouriensis",
        "group": "Actinobacteria",
    },
    {
        "id": 72,
        "accession": "NC_018524.1",
        "name": "Nocardiopsis dassonvillei",
        "group": "Actinobacteria",
    },
    {
        "id": 73,
        "accession": "NC_019673.1",
        "name": "Saccharothrix espanaensis",
        "group": "Actinobacteria",
    },
    {
        "id": 74,
        "accession": "NC_020133.1",
        "name": "Mycobacterium liflandii",
        "group": "Actinobacteria",
    },
    {
        "id": 75,
        "accession": "NC_021191.1",
        "name": "Actinoplanes sp. SE50/110",
        "group": "Actinobacteria",
    },
    # Archaea (25 genomes) - Third domain of life
    {
        "id": 76,
        "accession": "NC_000854.2",
        "name": "Aeropyrum pernix K1",
        "group": "Archaea",
    },
    {
        "id": 77,
        "accession": "NC_000868.1",
        "name": "Pyrococcus abyssi GE5",
        "group": "Archaea",
    },
    {
        "id": 78,
        "accession": "NC_000917.1",
        "name": "Archaeoglobus fulgidus DSM 4304",
        "group": "Archaea",
    },
    {
        "id": 79,
        "accession": "NC_002607.1",
        "name": "Halobacterium salinarum R1",
        "group": "Archaea",
    },
    {
        "id": 80,
        "accession": "NC_003552.1",
        "name": "Methanosarcina acetivorans C2A",
        "group": "Archaea",
    },
    {
        "id": 81,
        "accession": "NC_003106.2",
        "name": "Sulfolobus tokodaii str. 7",
        "group": "Archaea",
    },
    {
        "id": 82,
        "accession": "NC_003413.1",
        "name": "Pyrococcus furiosus DSM 3638",
        "group": "Archaea",
    },
    {
        "id": 83,
        "accession": "NC_007426.1",
        "name": "Natronomonas pharaonis",
        "group": "Archaea",
    },
    {
        "id": 84,
        "accession": "NC_008701.1",
        "name": "Pyrobaculum islandicum",
        "group": "Archaea",
    },
    {
        "id": 85,
        "accession": "NC_009073.1",
        "name": "Pyrobaculum calidifontis",
        "group": "Archaea",
    },
    {
        "id": 86,
        "accession": "NC_009776.1",
        "name": "Ignicoccus hospitalis",
        "group": "Archaea",
    },
    {
        "id": 87,
        "accession": "NC_009975.1",
        "name": "Methanococcus maripaludis C6",
        "group": "Archaea",
    },
    {
        "id": 88,
        "accession": "NC_010085.1",
        "name": "Nitrosopumilus maritimus",
        "group": "Archaea",
    },
    {
        "id": 89,
        "accession": "NC_013156.1",
        "name": "Methanocaldococcus fervens",
        "group": "Archaea",
    },
    {
        "id": 90,
        "accession": "NC_013407.1",
        "name": "Methanocaldococcus vulcanius",
        "group": "Archaea",
    },
    {
        "id": 91,
        "accession": "NC_014122.1",
        "name": "Methanocaldococcus infernus",
        "group": "Archaea",
    },
    {
        "id": 92,
        "accession": "NC_014222.1",
        "name": "Methanococcus voltae A3",
        "group": "Archaea",
    },
    {
        "id": 93,
        "accession": "NC_014804.1",
        "name": "Thermococcus barophilus",
        "group": "Archaea",
    },
    {
        "id": 94,
        "accession": "NC_015865.1",
        "name": "Thermococcus sp. 4557",
        "group": "Archaea",
    },
    {
        "id": 95,
        "accession": "NC_016051.1",
        "name": "Thermococcus onnurineus",
        "group": "Archaea",
    },
    {
        "id": 96,
        "accession": "NC_017275.1",
        "name": "Sulfolobus islandicus Y.N.15.51",
        "group": "Archaea",
    },
    {
        "id": 97,
        "accession": "NC_018092.1",
        "name": "Methanobrevibacter smithii",
        "group": "Archaea",
    },
    {
        "id": 98,
        "accession": "NC_019791.1",
        "name": "Caldisphaera lagunensis",
        "group": "Archaea",
    },
    {
        "id": 99,
        "accession": "NC_020156.1",
        "name": "Methanobacterium sp. AL-21",
        "group": "Archaea",
    },
    {
        "id": 100,
        "accession": "NC_021054.1",
        "name": "Haloferax volcanii DS2",
        "group": "Archaea",
    },
]


# Helper functions for the catalog
def get_genome_by_id(genome_id: int) -> dict:
    """Get genome info by catalog ID (1-100)"""
    for genome in GENOME_CATALOG:
        if genome["id"] == genome_id:
            return genome
    return None


def get_genome_by_accession(accession: str) -> dict:
    """Get genome info by NCBI accession"""
    for genome in GENOME_CATALOG:
        if genome["accession"] == accession:
            return genome
    return None


def list_genomes_by_group(group: str = None) -> list:
    """List genomes, optionally filtered by group"""
    if group:
        return [g for g in GENOME_CATALOG if g["group"] == group]
    return GENOME_CATALOG
