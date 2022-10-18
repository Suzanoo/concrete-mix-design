import os
import numpy as np

def save_to_multiple_csv_files(data, name_prefix, header=None, n_parts=10):
    new_dir = os.path.join(".", "new_data")
    os.makedirs(new_dir, exist_ok=True) 
    path_format = os.path.join(new_dir, "my_{}_{:02d}.csv")

    filepaths = []
    m = len(data) # nrows
    '''
    ex.
    row index : m = 11610
    array of row index : np.arange(m) = [1, 2, 3, ..., 11610]
    split to 10 arrays : np.array_split(np.arange(m), n_parts) = [[...], [...], ..., [...]]
    enumerate to access index and values for each
    '''
    for file_idx, row_indices in enumerate(np.array_split(np.arange(m), n_parts)):
        part_csv = path_format.format(name_prefix, file_idx)
        filepaths.append(part_csv)
        with open(part_csv, "wt", encoding="utf-8") as f:
            if header is not None:
                f.write(header)
                f.write("\n")
            for row_idx in row_indices:
                f.write(",".join([repr(col) for col in data[row_idx]]))
                f.write("\n")
    return filepaths
