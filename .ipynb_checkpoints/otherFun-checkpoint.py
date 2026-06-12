import numpy as np

def match_table_lengths(table1, table2, seed=None):
    """
    Randomly removes rows from the longer table so both tables have equal length.

    Parameters:
    -----------
    table1 : astropy.table.Table
        The first input table.
    table2 : astropy.table.Table
        The second input table.
    seed : int, optional
        Random seed for reproducibility.

    Returns:
    --------
    new_table1 : astropy.table.Table
        Table 1 after possible truncation.
    new_table2 : astropy.table.Table
        Table 2 after possible truncation.
    """
    if seed is not None:
        np.random.seed(seed)

    len1 = len(table1)
    len2 = len(table2)

    if len1 == len2:
        return table1, table2

    if len1 > len2:
        indices_to_keep = np.random.choice(len1, len2, replace=False)
        new_table1 = table1[sorted(indices_to_keep)]
        return new_table1, table2
    else:
        indices_to_keep = np.random.choice(len2, len1, replace=False)
        new_table2 = table2[sorted(indices_to_keep)]
        return table1, new_table2

def resolve_duplicates(table, keep_marker):
    # keep marker is 'bgs' and 'lrg'. this removes one of each duplicate and keeps version of the galaxy type you indicate
    is_dup_array = table['duplicate']
    non_duplicates = table[~is_dup_array]
    
    all_duplicates = table[is_dup_array]
    
    type_array = all_duplicates['galaxy_type']
    preferred_duplicates = all_duplicates[type_array == keep_marker]
    
    cleaned_table = vstack([non_duplicates, preferred_duplicates])
    
    return cleaned_table