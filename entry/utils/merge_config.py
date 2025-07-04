import sys
from omegaconf import OmegaConf

# Function to merge two configurations
def merge_configs(config_a, config_b, exception_keys):
    merged_config = OmegaConf.create()

    # Get all unique keys from both configurations
    all_keys = set(config_a.keys()).union(set(config_b.keys()))
    
    for key in all_keys:
        if key in exception_keys:  # Special handling for exception key
            merged_config[key] = config_a.get(key)
        else:
            merged_config[key] = config_b.get(key) if key in config_b else config_a.get(key)

    return merged_config

# Sample file paths for the YAML files
file_a = 'configs/pretrain.yaml'
file_b = sys.argv[1] 
output_file = file_b.replace("config.yaml", "merged.yaml")

# Exception key and its subfields
exception_keys = ['available_corpus', 'test_file', 'eval_corpus']

# Load the YAML files using OmegaConf
config_a = OmegaConf.load(file_a)
config_b = OmegaConf.load(file_b)

# Merge the configurations
merged_config = merge_configs(config_a, config_b, exception_keys)

# Save the merged configuration to a new file
OmegaConf.save(merged_config, output_file)