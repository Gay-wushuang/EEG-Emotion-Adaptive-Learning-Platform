RANDOM_SEED = 42

CSV_GROUPS = {
    'filtered': ['filtered.csv'],
    'powerspec': ['powerspec.csv'],
    'att': ['att.csv'],
    'med': ['med.csv']
}

TIME_STEPS = 10
TEST_RATIO = 0.2
NOISE_STD = 0.02
JITTER_RATIO = 0.03
MIXUP_ALPHA = 0.4

MODALITY_TYPES = {
    'filtered': 'signal',
    'powerspec': 'signal',
    'att': 'scalar',
    'med': 'scalar'
}
