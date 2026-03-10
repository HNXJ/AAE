import numpy as np
import pandas as pd

def get_binary_events_for_code(nwb_file, target_code=50.0, target_interval_name=None, code_column='codes'):
    """
    Extracts binary event indicators from an NWBFile object based on a target_code
    in a specified interval table.
    """
    if not hasattr(nwb_file, 'intervals') or not nwb_file.intervals:
        print("No interval tables found in the NWB file.")
        return np.array([])

    if not target_interval_name or target_interval_name not in nwb_file.intervals:
        print(f"Warning: Specified interval table '{target_interval_name}' not found or not provided. Returning empty array.")
        return np.array([])

    interval_table = nwb_file.intervals[target_interval_name]
    df = interval_table.to_dataframe()

    if code_column not in df.columns:
        print(f"Warning: Interval table '{target_interval_name}' does not contain '{code_column}' column. Returning array of zeros.")
        return np.zeros(len(df), dtype=int)

    if 'start_time' not in df.columns:
        print(f"Warning: Interval table '{target_interval_name}' does not contain 'start_time' column. This may affect interpretation.")

    # Convert the column to numeric, coercing errors to NaN
    codes_for_comparison = pd.to_numeric(df[code_column], errors='coerce')

    # Create a boolean mask using robust floating-point comparison
    binary_mask = np.isclose(codes_for_comparison, target_code, equal_nan=False)

    # Convert boolean mask to integer array (True -> 1, False -> 0)
    binary_array = binary_mask.astype(int)

    return binary_array
