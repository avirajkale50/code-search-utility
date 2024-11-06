import pandas as pd
from sentence_transformers import SentenceTransformer
import numpy as np

# Function to generate embeddings from the CSV file
def generate_embeddings(df):
    # Rename columns to match the expected case
    df.columns = df.columns.str.strip()  # Remove any leading/trailing spaces
    df.rename(columns={'instruction': 'Instruction', 'input': 'Input', 'output': 'Output'}, inplace=True)

    # Extract the relevant columns from the DataFrame
    instructions = df['Instruction'].tolist()  # Ensure this matches the column name
    inputs = df['Input'].tolist()
    outputs = df['Output'].tolist()

    # Handle missing or non-string values (NaN, floats, etc.)
    instructions = [str(i) if isinstance(i, str) else str(i) for i in instructions]
    inputs = [str(i) if isinstance(i, str) else str(i) for i in inputs]
    outputs = [str(o) if isinstance(o, str) else str(o) for o in outputs]

    # Initialize Sentence Transformer model
    model = SentenceTransformer('paraphrase-MiniLM-L6-v2')  # Example model, adjust as necessary

    # Generate embeddings for each of the columns
    instruction_embeddings = model.encode(instructions)  # Embedding for Instructions
    input_embeddings = model.encode(inputs)  # Embedding for Inputs
    output_embeddings = model.encode(outputs)  # Embedding for Outputs

    # Optionally, return embeddings as a dictionary
    return {
        'instructions': instruction_embeddings,
        'inputs': input_embeddings,
        'outputs': output_embeddings
    }

# Main function to execute the process
def main():
    # Read the CSV file (change this to your actual file path)
    df = pd.read_csv('data/train.csv')  # Make sure to change this path to your file

    # Print the columns to ensure they match
    print("CSV Columns:", df.columns)

    # Check if all required columns are present
    required_columns = ['Instruction', 'Input', 'Output']
    if not all(col in df.columns for col in required_columns):
        print(f"Error: The CSV file must contain the following columns: {', '.join(required_columns)}")
        return

    # Generate embeddings for the data
    embeddings = generate_embeddings(df)

    # Save the embeddings as .npy files (can be loaded later for fast access)
    np.save('embeddings/instruction_embeddings.npy', embeddings['instructions'])
    np.save('embeddings/input_embeddings.npy', embeddings['inputs'])
    np.save('embeddings/output_embeddings.npy', embeddings['outputs'])

    print("Embeddings have been generated and saved successfully.")

# Run the main function
if __name__ == '__main__':
    main()
