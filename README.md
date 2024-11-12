# Code Search Utility

A powerful code search utility that uses modern NLP techniques and embeddings to find and rank similar code snippets based on natural language queries. This project leverages sentence transformers for embedding generation and cosine similarity for matching relevant code examples.

## 🚀 Features

- **Smart Code Search**: Find relevant code snippets using natural language queries
- **Semantic Understanding**: Utilizes sentence transformers to understand the meaning behind code and queries
- **Similarity Ranking**: Returns results ranked by relevance using cosine similarity
- **Interactive Web Interface**: Built with Streamlit for easy interaction
- **Efficient Data Processing**: Handles CSV data with pandas for code snippet management

## 🛠️ Technical Architecture

1. **Data Preprocessing**
   - CSV file loading and cleaning
   - Handles code snippets with instructions, inputs, and outputs

2. **Feature Extraction**
   - Generates embeddings using sentence transformers
   - Creates vector representations for both inputs and outputs

3. **Similarity Measurement**
   - Implements cosine similarity for comparing query and code snippets
   - Ranks results based on similarity scores

4. **Search and Ranking System**
   - Real-time query processing
   - Returns most relevant code snippets

## 📋 Prerequisites

```bash
pip install pandas
pip install sentence-transformers
pip install scikit-learn
pip install streamlit
```

## 🚀 Getting Started

1. Clone the repository:
```bash
git clone https://github.com/yourusername/code-search-engine.git
cd code-search-engine
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Prepare your data:
   - Place your code snippets CSV file in the `data` directory
   - Ensure your CSV has the required columns (Instruction, Input, Output)

4. Run the Streamlit app:
```bash
streamlit run app.py
```

## 💻 Usage

1. Open the web interface in your browser
2. Enter your query in natural language or code
3. View ranked results of similar code snippets
4. Copy or download the code you need

## 🔍 How It Works

The system uses a combination of data mining and NLP techniques:

- **Vector Space Model (VSM)**: Represents code snippets as vectors in high-dimensional space
- **Sentence Transformers**: Generates meaningful embeddings from code and queries
- **Cosine Similarity**: Measures similarity between query and code snippets
- **Nearest Neighbor Search**: Finds the most relevant code examples

## 🛠️ Technical Components

```python
# Example: Loading and processing data
df = pd.read_csv('data/your_code_data.csv')

# Generate embeddings
input_embeddings = model.encode(inputs)
output_embeddings = model.encode(outputs)

# Calculate similarity
similarities = cosine_similarity(query_embedding, embeddings)
ranked_results = sorted(enumerate(similarities[0]), 
                       key=lambda x: x[1], 
                       reverse=True)
```

## 🔧 Optional Extensions

- Topic Modeling for code categorization
- Association Rule Mining for related code snippets
- Deep Learning Models for enhanced understanding
- Clustering for better organization of code snippets

## 📈 Future Improvements

- [ ] Add support for multiple programming languages
- [ ] Implement code syntax highlighting
- [ ] Add user authentication
- [ ] Include code execution environment
- [ ] Add support for code snippet versioning

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 📧 Contact

For any questions or feedback, please open an issue in the GitHub repository.
