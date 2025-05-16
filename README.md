# RAG vs Long Context Demonstration with Django Frontend

This project demonstrates the difference between RAG (Retrieval-Augmented Generation) and Long Context approaches in natural language processing, with a modern web interface built using Django.

## Key Differences

1. **RAG Approach**:
   - Uses a vector database (FAISS) to store and retrieve relevant documents
   - Combines external knowledge with language model generation
   - Can access external knowledge beyond the model's context window
   - More accurate for specific domain knowledge

2. **Long Context Approach**:
   - Uses a single large context window
   - All information must be provided within the prompt
   - Limited by the model's maximum context size
   - More straightforward to implement

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Set your OpenAI API key:
```bash
set OPENAI_API_KEY=your_api_key_here
```

3. Run the Django development server:
```bash
python manage.py runserver
```

4. Open your browser and navigate to:
```
http://127.0.0.1:8000/
```

## Features

- Modern, responsive UI using Bootstrap 4
- Side-by-side comparison of RAG and Long Context approaches
- Interactive question input
- Clear display of results for both approaches
- Hover effects and card styling for better visual appeal

## What You'll See

The code will demonstrate both approaches by:
1. Creating sample documents about RAG and Long Context
2. Using RAG to retrieve relevant documents and generate an answer
3. Using Long Context to generate an answer from a single large prompt

The web interface will:
1. Show a side-by-side comparison of RAG and Long Context approaches
2. Display the default comparison when first loaded
3. Allow you to enter your own questions
4. Show how each approach handles the same question differently

The UI includes:
- Two cards showing RAG and Long Context approaches
- Question input box at the bottom
- Beautiful styling with hover effects
- Responsive design that works on all screen sizes
