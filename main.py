import yaml
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.load_evaluations import load_evaluation

if len(sys.argv) > 1:
    configfname = sys.argv[1]
else:
    configfname = 'config.yaml'

with open(configfname, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

exp = load_evaluation(config)
exp.run()
