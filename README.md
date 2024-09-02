# Unity BERT Question Answering Integration

This Unity project integrates the DistilBERT ONNX model using Unity Sentis for question-answering tasks. The project includes three scripts: a tokenizer, a tokenizer test script, and the main question-answering script. The integration demonstrates how to tokenize input, run the model, and extract answers from logits.

## Features

- **Tokenizer Script**: Handles text tokenization using a vocabulary file.
- **Tokenizer Test Script**: Tests the tokenizer by converting text to token IDs and decoding them back.
- **Main Script**: Loads the ONNX model, processes questions and context, and extracts answers based on logits.

## Scripts Overview

### Tokenizer Script: `BERTTokenizer.cs`

- Located in the `Assets` folder.
- Loads vocabulary from `vocab.txt` and tokenizes input text.
- Handles unknown tokens and basic WordPiece tokenization.
- Includes decoding functionality to convert token IDs back to text.

### Tokenizer Test Script: `TokenizerTest.cs`

- Located in the `Assets` folder.
- Initializes the tokenizer with `vocab.txt`.
- Tests tokenization and decoding functions with sample text.
- Outputs results to the Unity Console.

### Main Script: `BERTQuestionAnswering.cs`

- Located in the `Assets` folder.
- Loads the DistilBERT ONNX model from `Resources/model-4.onnx`.
- Processes input question and context to generate answers using logits.
- Uses `Unity.Sentis` for model inference on GPU.
- Cleans up resources after execution.

## Installation

1. **Clone the Repository**:

   ```bash
   git clone https://github.com/jaypatel73/Distilbert_Unity_Integration.git
