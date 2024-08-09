import yaml
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.load_evaluations import load_evaluation
from src.utils.load_trainers import load_trainer

if len(sys.argv) > 1:
    configfname = sys.argv[1]
else:
    configfname = 'config.yaml'

with open(configfname, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

modes = config['mode']
for mode in modes:
    if mode == 'interact':
        exp = load_evaluation(config)
        exp.interact()
    if mode == 'evaluate':
        exp = load_evaluation(config)
        exp.evaluate()
    if mode == 'train':
        exp = load_trainer(config)
        exp.train()
