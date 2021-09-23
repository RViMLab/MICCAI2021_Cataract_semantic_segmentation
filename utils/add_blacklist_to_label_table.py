import pathlib
import pandas as pd
import numpy as np

label_table = pd.read_pickle(pathlib.Path('../data/label_table.pkl'), compression='gzip')
data = pd.read_csv(pathlib.Path('../data/data.csv'))

label_table['blacklisted'] = data['blacklisted']
label_table['img_path'] = data['img_path']

for ind, row in label_table.iterrows():

    name = row['file_name']
    path = row['img_path']
    assert(name in path), 'ind {} name {} not in path {}'.format(ind, name, path)

label_table = label_table.drop('img_path', axis=1)
label_table.to_pickle(pathlib.Path('../data/label_table_with_blacklist.pkl'), compression='gzip')
label_table.to_csv(pathlib.Path('../data/label_table_with_blacklist.csv'))
