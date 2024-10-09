import pandas as pd 
import yaml
import sys

#configfname = sys.argv[1]
configfname = 'clams/config.yaml'
sys.stderr.write(f'Reading from {configfname}\n')

with open(configfname, 'r') as f:
    config = yaml.load(f, Loader=yaml.FullLoader)


data_with_conditions = pd.read_csv(config['datafpath'], sep='\t')
evaluate_results = pd.read_csv(config['predfpath'], sep='\t')
print(data_with_conditions)
print(evaluate_reults)
