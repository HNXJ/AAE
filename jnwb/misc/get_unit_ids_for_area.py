import pandas as pd

def get_unit_ids_for_area(nwb_file, target_area_name):
    """
    Retrieves the IDs of units located in a specified brain area from an NWBFile.
    """
    if nwb_file.units is None or len(nwb_file.units) == 0:
        return []

    if nwb_file.electrodes is None or len(nwb_file.electrodes) == 0:
        return []

    units_df = nwb_file.units.to_dataframe().reset_index()
    electrodes_df = nwb_file.electrodes.to_dataframe().reset_index()

    if 'peak_channel_id' not in units_df.columns:
        return []
    if 'id' not in electrodes_df.columns:
        return []

    area_col_name = None
    if 'location' in electrodes_df.columns:
        area_col_name = 'location'
    elif 'label' in electrodes_df.columns:
        area_col_name = 'label'
    else:
        return []

    units_df['peak_channel_id'] = units_df['peak_channel_id'].astype(float).astype(int)
    electrodes_df['id'] = electrodes_df['id'].astype(int)

    merged_df = pd.merge(
        units_df,
        electrodes_df[[area_col_name, 'id']],
        left_on='peak_channel_id',
        right_on='id',
        how='left',
        suffixes=('_unit', '_electrode')
    )

    merged_df['area_normalized'] = merged_df[area_col_name].apply(lambda x: x.decode('utf-8').strip().upper() if isinstance(x, bytes) else str(x).strip().upper())
    target_area_name_normalized = target_area_name.strip().upper()

    filtered_units = merged_df[merged_df['area_normalized'] == target_area_name_normalized]

    return filtered_units['id_unit'].tolist()
