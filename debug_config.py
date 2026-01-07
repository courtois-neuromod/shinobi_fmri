import yaml
import os

config_path = "config.yaml"
print(f"Loading config from: {os.path.abspath(config_path)}")

with open(config_path, 'r') as f:
    try:
        data = yaml.safe_load(f)
        print("Config loaded successfully.")
    except yaml.YAMLError as exc:
        print(f"Error loading yaml: {exc}")
        exit(1)

if 'analysis' in data:
    print("Found 'analysis' section.")
    keys = data['analysis'].keys()
    print(f"Keys in 'analysis': {list(keys)}")
    
    if 'low_level_conditions' in data['analysis']:
        print(f"Found 'low_level_conditions': {data['analysis']['low_level_conditions']}")
    else:
        print("ERROR: 'low_level_conditions' NOT FOUND in 'analysis' section.")
else:
    print("ERROR: 'analysis' section NOT FOUND.")
