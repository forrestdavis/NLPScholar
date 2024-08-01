import yaml
from src.experiments.TSE import TSE
from src.experiments.Interact import Interact

configfname = 'config.yaml'
with open(configfname, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)

if config['exp'] == 'Interact': 
    exp_cls = Interact
elif config['exp'] == 'TSE':
    exp_cls = TSE

exp = exp_cls(config)
exp.run()
