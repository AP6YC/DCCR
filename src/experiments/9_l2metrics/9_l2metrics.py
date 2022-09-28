from l2metrics import l2metrics
from l2logger import l2logger

dir = 'work/results/9_l2metrics'
name = 'test_scenario'
cols = {'metrics_columns': ['accuracy']}
meta = {
    'author': 'Sasha Petrenko',
    'complexity': '1-low',
    'difficulty': '2-medium',
    'scenario_type': 'custom',
    'scenario_version': '0.1'
}
logger = l2logger.DataLogger(dir, name, cols, meta)
