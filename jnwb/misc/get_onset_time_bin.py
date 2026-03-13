def get_onset_time_bin(nwb_file, binary_event_array, target_interval_name):
    """
    Retrieves start_time values from the specified interval table based on a binary event array.
    """
    onset_times = []

    if not hasattr(nwb_file, 'intervals') or not nwb_file.intervals:
        print("No interval tables found in the NWB file.")
        return onset_times

    if not target_interval_name or target_interval_name not in nwb_file.intervals:
        print(f"Warning: Specified interval table '{target_interval_name}' not found or not provided. Returning empty list.")
        return onset_times

    interval_table = nwb_file.intervals[target_interval_name]
    df = interval_table.to_dataframe()

    if 'start_time' not in df.columns:
        print(f"Warning: Interval table '{target_interval_name}' does not contain 'start_time' column. Returning empty list.")
        return onset_times

    # Ensure binary_event_array matches the length of the DataFrame
    if len(binary_event_array) != len(df):
        print("Error: Length of binary_event_array does not match the length of the interval table. Returning empty list.")
        return onset_times

    # Filter start_times where the binary_event_array is 1
    onset_times = df['start_time'][binary_event_array == 1].tolist()

    return onset_times
