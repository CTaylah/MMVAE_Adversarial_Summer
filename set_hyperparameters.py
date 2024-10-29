import yaml
import argparse
import os
import subprocess

def edit_yaml(file_path, changes):
    # Read the YAML file
    with open(file_path, 'r') as file:
        data = yaml.safe_load(file)

    # Apply the changes
    for key, value in changes.items():
        keys = key.split('.')
        d = data
        for k in keys[:-1]:
            d = d[k]
        d[keys[-1]] = value

    # Return updated dictionary
    return data

def parse_args():
    parser = argparse.ArgumentParser(description='Edit YAML configuration file.')
    parser.add_argument('--key', type=str, required=True, help='Key to be changed')
    parser.add_argument('--values', type=str, required=True, help='Values to be set for the key')
    return parser.parse_args()

def create_experiments_folder(hyperparam: str):
    new_folder = "/active/debruinz_project/cardell_taylor/MMVAE/experiments/" + hyperparam
    os.makedirs(new_folder, exist_ok=True)
    return new_folder


#hyper_param = {hyperparm, value}
def create_experiements_yaml(hyperparam: str, value, model_path: str, experiment_path: str):
    changes = {"experiment_name": hyperparam,
               "run_name": "value=" + str(value),
               "train_command.fit.model": model_path}
    base_path = "/active/debruinz_project/cardell_taylor/MMVAE/experiments.yaml"
    data = edit_yaml(base_path, changes)

    path = os.path.join(experiment_path, str(value) + ".yaml")
    with open(path, 'w') as file:
        yaml.safe_dump(data, file, sort_keys=False)

    print(f"Updated YAML file saved to {path}")
    return path
    
def create_model_config(hyperparam: str, value, experiment_path: str):
    changes = {hyperparam: value}
    config_path = "/active/debruinz_project/cardell_taylor/MMVAE/configs/model/config.yaml"
    data = edit_yaml(config_path, changes)

    new_folder = experiment_path + "/model/"
    os.makedirs(new_folder, exist_ok=True)
    path = os.path.join(new_folder, str(value) + ".yaml")
    with open(path, 'w') as file:
        yaml.safe_dump(data, file, sort_keys=False)

    print(f"Updated YAML file saved to {path}")
    return path

def main():
    args = parse_args()

    # Parse changes from command line arguments
    key = args.key
    values = args.values

    # Convert values to appropriate types
    try:
        true_values = eval(values)
    except:
        print(f"Values {values} could not be evaluated.")
        true_values = values  # Add the raw value if it cannot be evaluated

    changes = {key: true_values}
    print(changes)

    for value in true_values:
        folder = create_experiments_folder(key)
        model_path = create_model_config(key, value, folder)
        experiment_path = create_experiements_yaml(key, value, model_path, folder)
        command = ['cmmvae', 'submit', '-t', '--config_file', experiment_path] 
        result = subprocess.run(command, capture_output=True, text=True)
        print(result.stdout)

if __name__ == '__main__':
    main()