import pandas as pd
import os

def csv_chunk_splitter(input_file, output_path, chunk_size, max_size=10*1024**3):
    """
    Split a large CSV file into multiple smaller files.
    :param input_file: Path to the input CSV file.
    :param output_path: Directory to store the split CSV files.
    :param chunk_size: Number of rows per chunk.
    :param max_size: Maximum size of each split file in bytes (default is 10GB).
    """
    chunk_no = 1
    current_size = 0
    for chunk in pd.read_csv(input_file, chunksize=chunk_size):
        output_file = os.path.join(output_path, f'split_{chunk_no}.csv')
        
        # Append to the current file if it's not too big
        if current_size < max_size:
            chunk.to_csv(output_file, mode='a', index=False, header=not os.path.exists(output_file))
            current_size += chunk.memory_usage(index=True, deep=True).sum()
        else:
            # Start a new file
            chunk_no += 1
            current_size = chunk.memory_usage(index=True, deep=True).sum()
            chunk.to_csv(output_file, mode='w', index=False)

input_csv = 'Taxi_Trips_20231029.csv'
output_directory = '.'
chunk_size = 10000000 # Adjust this based on your memory capacity

csv_chunk_splitter(input_csv, output_directory, chunk_size)