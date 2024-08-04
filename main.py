import yaml
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent))
from src.utils.load_experiments import load_experiment

if len(sys.argv) > 1:
    configfname = sys.argv[1]
else:
    configfname = 'config.yaml'

with open(configfname, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

exp = load_experiment(config)
exp.run()
