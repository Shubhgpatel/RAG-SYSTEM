# 🚀 Enhanced RAG System with Ollama

Advanced Retrieval-Augmented Generation system with semantic search, document processing, and local LLM integration.

![Python](https://img.shields.io/badge/python-v3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/streamlit-latest-red.svg)

## ✨ Features

- **Semantic Search**: Advanced embeddings with ChromaDB vector storage
- **Multi-Format Support**: PDF, DOCX, PPTX, Excel, Images (OCR), TXT
- **Source Citations**: See exactly where answers come from
- **Cross-Encoder Reranking**: Improved result relevance
- **Local LLM**: Ollama integration for privacy
- **Persistent Storage**: Your data stays between sessions

## 🛠️ Setup Guide

### Prerequisites
- Python 3.8+ installed
- At least 4GB RAM
- 2GB free disk space

### Step 1: Clone Repository
```bash
git clone https://github.com/yourusername/enhanced-rag-system.git
cd enhanced-rag-system
```

### Step 2: Install Python Dependencies
```bash
# Create virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# or
venv\Scripts\activate     # Windows

# Install all required packages
pip install streamlit PyPDF2 sentence-transformers chromadb requests python-docx python-pptx pandas Pillow pytesseract langchain openpyxl xlrd
```

### Step 3: Install Ollama

**Linux/Mac:**
```bash
curl -fsSL https://ollama.ai/install.sh | sh
```

**Windows:**
Download from [ollama.ai](https://ollama.ai/download) and install

### Step 4: Setup Ollama Model
```bash
# Download recommended model (3.8GB)
ollama pull llama2

# For faster responses (smaller model)
ollama pull llama2:7b

# For better quality (larger model)
ollama pull llama2:13b
```

### Step 5: Install Tesseract (for OCR)

**Ubuntu/Debian:**
```bash
sudo apt-get update
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

**Windows:**
1. Download from [GitHub](https://github.com/UB-Mannheim/tesseract/wiki)
2. Install and add to PATH

### Step 6: Start the System
```bash
# Terminal 1: Start Ollama server
ollama serve

# Terminal 2: Start the RAG application
streamlit run enhanced_rag.py
```

The app will open at `http://localhost:8501`

## 📖 How to Use

### Initial Setup
1. **Start Ollama**: Ensure Ollama is running (`ollama serve`)
2. **Open the App**: Navigate to `http://localhost:8501`
3. **Test Connection**: 
   - Go to sidebar → "🦙 Ollama Settings"
   - Click "🔌 Test Connection"
   - Should show "✅ Connected!"

### Document Management
1. **Upload Documents**:
   - Click "Browse files" in sidebar
   - Select multiple files (PDF, DOCX, PPTX, Excel, Images, TXT)
   - Wait for "✅ Added [filename]" confirmation

2. **Supported Formats**:
   - **PDF**: Research papers, reports, manuals
   - **Word**: Documents, contracts, notes
   - **PowerPoint**: Presentations, slides
   - **Excel**: Spreadsheets, data tables
   - **Images**: Screenshots, scanned documents
   - **Text**: Plain text files, code files

### Chatting with Documents
1. **Ask Questions**:
   ```
   Examples:
   • "What are the main conclusions?"
   • "Summarize the methodology section"
   • "What data sources were used?"
   • "Compare findings from document A and B"
   ```

2. **Advanced Queries**:
   ```
   • "What are the key findings? Also, what are the limitations?"
   • "Explain the methodology and list all data sources mentioned"
   • "What recommendations does each document provide?"
   ```

3. **View Results**:
   - Read the AI-generated answer
   - Expand "📚 Sources Used" to see:
     - Which documents were referenced
     - Relevance scores (🟢 High, 🟡 Medium, 🔴 Low)
     - Text snippets that supported the answer

### Tips for Better Results
1. **Upload Related Documents**: Better context for comprehensive answers
2. **Be Specific**: "What were the 2023 sales figures?" vs "Tell me about sales"
3. **Ask Follow-ups**: Build on previous questions for deeper insights
4. **Check Sources**: Verify information using the source citations
5. **Use Natural Language**: Write questions as you would ask a colleague

### Configuration Options
1. **Model Selection**:
   - Sidebar → "Select Model"
   - Choose based on your needs:
     - `llama2:7b` - Fast responses
     - `llama2:13b` - Better quality
     - `mistral` - Good balance

2. **Query Enhancement**:
   - Check "🔍 Enhance query" for better search results
   - Automatically expands your question for comprehensive answers

### Managing Your Data
1. **View Loaded Documents**:
   - Check "📋 Loaded Documents" in sidebar
   - See chunk count and preview for each file

2. **Clear All Data**:
   - Click "🗑️ Clear All Documents"
   - Removes all uploaded files and chat history

3. **Persistent Storage**:
   - Your documents persist between sessions
   - Stored in `./chroma_db/` folder
   - Chat history resets each session

## 🛠️ Configuration

### Environment Variables
```env
OLLAMA_BASE_URL=http://localhost:11434
OLLAMA_MODEL=llama2
CHUNK_SIZE=500
```

### Recommended Models
- **General**: `llama2`, `mistral`
- **Code**: `codellama`
- **Large docs**: `llama2:13b`

## 🔧 Troubleshooting

**Ollama connection failed?**
```bash
# Check if running
curl http://localhost:11434/api/tags
ollama serve
```

**Import errors?**
```bash
pip install --upgrade sentence-transformers chromadb
```

**OCR not working?**
```bash
# Ubuntu/Debian
sudo apt-get install tesseract-ocr

# macOS
brew install tesseract
```

## 📁 Project Structure

```
enhanced-rag-system/
├── enhanced_rag.py     # Main application
├── requirements.txt    # Dependencies
├── chroma_db/         # Vector database (auto-created)
└── cache/             # Embedding cache (auto-created)
```

## 🤝 Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature-name`
3. Make changes and test
4. Submit pull request


## 🙏 Acknowledgments

Built with Streamlit, Ollama, ChromaDB, and Sentence Transformers.

---

⭐ **Star this repo if it helped you!**
