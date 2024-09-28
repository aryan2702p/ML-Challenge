import pandas as pd
import os

def shuffle_and_split_csv(file_path, batch_size):
    # Read the CSV file into a DataFrame
    df = pd.read_csv(file_path,encoding='latin1')
    
    # Shuffle the DataFrame
    df = df.sample(frac=1).reset_index(drop=True)

    # Calculate the number of batches needed
    total_rows = len(df)
    num_batches = (total_rows // batch_size) + (1 if total_rows % batch_size != 0 else 0)

    # Create a directory for the output files if it doesn't exist
    output_dir = "output_batches"
    os.makedirs(output_dir, exist_ok=True)

    # Split the DataFrame and save each batch as a new CSV file
    for i in range(num_batches):
        start_row = i * batch_size
        end_row = start_row + batch_size
        batch_df = df.iloc[start_row:end_row]
        
        # Define the output file name
        output_file = os.path.join(output_dir, f'batch_{i + 1}.csv')
        
        # Save the batch DataFrame to a new CSV file
        batch_df.to_csv(output_file, index=False)
        print(f'Saved: {output_file}')

# Example usage
file_path = 'dataset/train.csv'  # Replace with your actual CSV file path
batch_size = 70000            # Define your desired batch size
shuffle_and_split_csv(file_path, batch_size)