import argparse
import pandas as pd
import pickle

def get_unique_labels(file_path, column_name):
    my_set = set()
    for i in range(15):
        chunk = i + 1
        human_path = file_path + "human_metadata_" + str(chunk) + ".pkl"
        mouse_path = file_path + "mouse_metadata_" + str(chunk) + ".pkl"

        mouse_df = pd.read_pickle(mouse_path) 
        human_df = pd.read_pickle(human_path) 

        m_list = mouse_df[column_name].unique().tolist()
        h_list = human_df[column_name].unique().tolist()

        my_set.update(list(m_list))
        my_set.update(list(h_list))

    return list(my_set)
    

def create_dictionary_file(unique_entries: list, label: str):
    with open("/mnt/projects/debruinz_project/cardell_taylor/test_backprop/MMVAE_Adversarial_Summer/src/cmmvae/data/encoding_dicts/" + label + "_dict.py", "w") as file:
        file.write(label + " = { \n")
        for i in range(len(unique_entries)):
            file.writelines('"' + unique_entries[i] + '"'  + ": " + str(i) + ", \n")
        file.write("} \n")


parser = argparse.ArgumentParser(description='get the integer encoding of')
parser.add_argument('--file_path', type=str, required=True, help='to the directory')
parser.add_argument('--label', type=str, required=True, help='Name of the column. ex: assay')


file_path = parser.parse_args().file_path
label = parser.parse_args().label

unique_labels = get_unique_labels(file_path, label)
create_dictionary_file(unique_labels, label)