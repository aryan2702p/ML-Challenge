import pandas as pd
import os



# Loop through each file, read them, and combine
combined_data = []
# Read the original CSV file
original_df = pd.read_csv("dataset/test.csv", encoding='latin1')

# Read the processed CSV file
processed_df = pd.read_csv("test.csv")

# Combine the 'processed_text' from processed_df with 'target' from original_df
combined_df = pd.DataFrame({
    'processed_text': processed_df['processed_text'],
    'Index': original_df['Index']
})

# Append to combined_data list
combined_data.append(combined_df)

# Concatenate all combined DataFrames into one
final_combined_df = pd.concat(combined_data, ignore_index=True)

# Save the final combined DataFrame to a new CSV file
final_combined_df.to_csv('combined_output_test.csv', index=False)

print("Combined data saved to 'combined_output.csv'.")