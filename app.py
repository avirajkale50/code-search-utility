import streamlit as st
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load pre-computed embeddings (make sure the paths are correct)
instruction_embeddings = np.load('embeddings/instruction_embeddings.npy')
input_embeddings = np.load('embeddings/input_embeddings.npy')
output_embeddings = np.load('embeddings/output_embeddings.npy')

# Load the original code snippets (instructions, inputs, and outputs) from the CSV
import pandas as pd
df = pd.read_csv('data/train.csv')
instructions = df['Instruction'].tolist()
inputs = df['Input'].tolist()
outputs = df['Output'].tolist()

# Initialize the SentenceTransformer model (use the same one you used for generating embeddings)
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Streamlit UI for input
st.title("Code Search Utility")

st.write(
    "Enter a query, and I'll help you find the most relevant code snippet."
)

# Text input for the user query
user_query = st.text_input("Enter your query here:")

# Function to find the most relevant code snippet using cosine similarity
def get_most_relevant_code(user_query):
    # Generate embedding for the user query
    query_embedding = model.encode([user_query])

    # Calculate cosine similarity between the user query and the precomputed embeddings
    instruction_similarities = cosine_similarity(query_embedding, instruction_embeddings)
    input_similarities = cosine_similarity(query_embedding, input_embeddings)
    output_similarities = cosine_similarity(query_embedding, output_embeddings)

    # Find the index of the highest similarity for each type (instruction, input, output)
    best_instruction_idx = np.argmax(instruction_similarities)
    best_input_idx = np.argmax(input_similarities)
    best_output_idx = np.argmax(output_similarities)

    # Prepare the results to return
    results = {
        "Best Instruction": instructions[best_instruction_idx],
        "Best Input": inputs[best_input_idx],
        "Best Output": outputs[best_output_idx],
    }
    
    return results

# When the user submits a query
if user_query:
    results = get_most_relevant_code(user_query)

    st.subheader("Most Relevant Code Snippets:")

    # Display the relevant code snippets
    st.write(f"**Instruction:** {results['Best Instruction']}")
    st.write(f"**Input:** {results['Best Input']}")
    st.markdown(f"### Output\n```python\n{results['Best Output']}\n```")

    st.download_button(
    label="Download Best Output as Python Code",
    data=results['Best Output'],
    file_name="best_output.py",  # Name of the file to be downloaded
    mime="text/python",  # MIME type for Python files
)
