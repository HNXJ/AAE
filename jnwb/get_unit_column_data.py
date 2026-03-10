def get_unit_column_data(nwb_file, column_label):
    """
    Retrieves data from a specified column of the nwb.units table.
    """
    if nwb_file.units is None or len(nwb_file.units) == 0:
        print("Units table is empty or does not exist in the NWB file.")
        return None

    if column_label not in nwb_file.units.colnames:
        print(f"Column '{column_label}' not found in the units table. Available columns: {nwb_file.units.colnames}")
        return None

    df_units = nwb_file.units.to_dataframe()
    return df_units[column_label]
