# Harel Insurance Customer Support Chatbot

> **Production-grade, domain-specific GenAI system for Israel's largest insurance provider**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.109+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

## 🎯 Project Goal

Build an end-to-end GenAI system that:
- ✅ Answers customer questions across **8 insurance domains**
- ✅ Grounds every answer in official documentation with **explicit citations**
- ✅ **Outperforms GPT-5 baseline** using retrieval and agentic design
- ✅ Achieves **<5% hallucination rate** and **>90% citation accuracy**

**Domains**: Car, Life, Travel, Health, Dental, Mortgage, Business, Apartment

---

## 📋 Table of Contents

- [Quick Start](#-quick-start)
- [Architecture](#-architecture)
- [Project Structure](#-project-structure)
- [Development Stages](#-development-stages)
- [Tech Stack](#-tech-stack)
- [Evaluation Metrics](#-evaluation-metrics)
- [Team](#-team)
- [Documentation](#-documentation)

---

## 🚀 Quick Start

### Prerequisites

- Python 3.10+
- Docker (for Milvus)
- OpenAI API key
- 16GB+ RAM recommended

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd harel-insurance-chatbot

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your API keys
```

### Run Milvus (Vector Database)

```bash
# Using Docker Compose
docker-compose up -d milvus

# Verify Milvus is running
curl http://localhost:19530/healthz
```

### Run the Application

```bash
# Start the FastAPI server
uvicorn src.api.main:app --reload --port 8000

# Access API documentation
open http://localhost:8000/docs
```

---

## 🏗️ Architecture

### 5-Agent Role-Based System

```
User Query
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Agent 1: Query Understanding & Decomposition                │
│ • Breaks down complex queries into sub-questions            │
│ • Identifies domains and intent                             │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Agent 2: Multi-Strategy Retrieval                           │
│ • Semantic search (dense vectors)                           │
│ • Keyword search (BM25 sparse)                              │
│ • Hybrid search (weighted combination)                      │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Agent 3: Context Ranking & Filtering                        │
│ • Cross-encoder re-ranking                                  │
│ • Deduplication and diversity sampling (MMR)                │
│ • Context compression                                       │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Agent 4: Answer Generation & Citation                       │
│ • Grounded generation (Qwen 2.5 72B)                        │
│ • Inline citation attachment                                │
│ • Multi-source synthesis                                    │
└─────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────────────────────────────────────────────────┐
│ Agent 5: Verification & Quality Assurance                   │
│ • Citation validation                                       │
│ • Hallucination detection                                   │
│ • Completeness check                                        │
└─────────────────────────────────────────────────────────────┘
    ↓
Grounded Answer + Citations
```

**Key Components**:
1. **5-Agent Pipeline**: Role-based agents with specialized skills (not domain-based)
2. **Multi-Strategy Retrieval**: Semantic + Keyword + Hybrid search with cross-encoder re-ranking
3. **Document-Type Aware Chunking**: 1024/768/512/256 tokens based on document type
4. **Verification Layer**: Multi-stage quality assurance to catch errors
5. **Model-Agnostic Design**: Easy to swap LLMs (Qwen, GPT, Claude)

**Why This Architecture Wins**:
- ✅ **Query Decomposition**: Handles complex multi-domain questions better than single-shot retrieval
- ✅ **Multi-Strategy Retrieval**: Hybrid search (semantic + keyword) beats pure vector search
- ✅ **Cross-Encoder Re-ranking**: Significantly improves relevance (65% of score!)
- ✅ **Verification Layer**: Catches hallucinations and citation errors before submission
- ✅ **Performance Optimizations**: Parallel retrieval, streaming, caching, speculative execution

See [DESIGN.md](DESIGN.md) for detailed architecture.

---

## 📁 Project Structure

```
harel-insurance-chatbot/
├── data/                       # Data storage
│   ├── raw/                    # Scraped ASPX and PDF files
│   ├── processed/              # Parsed and chunked documents
│   ├── embeddings/             # Generated embeddings
│   └── evaluation/             # Test sets and reference questions
├── src/                        # Source code
│   ├── scraping/               # Web scraping modules
│   ├── processing/             # Document processing (Docling)
│   ├── retrieval/              # RAG components (Milvus)
│   ├── agents/                 # Agent implementations (LangChain)
│   ├── generation/             # Answer generation and prompts
│   ├── evaluation/             # Evaluation framework (RAGAS)
│   └── api/                    # FastAPI application
├── tests/                      # Unit and integration tests
├── notebooks/                  # Jupyter notebooks for exploration
├── config/                     # Configuration files
├── scripts/                    # Utility scripts
├── requirements.txt            # Python dependencies
├── docker-compose.yml          # Docker services (Milvus)
├── .env.example                # Environment variables template
├── README.md                   # This file
└── DESIGN.md                   # Detailed system design
```

---

## 🎬 Development Stages

### Stage 1: Model Baseline & Evaluation (Week 1)
- [x] Scrape Harel insurance data
- [x] Build evaluation framework
- [x] Run GPT-4o and GPT-5.2 baselines
- [x] Generate baseline report

### Stage 2: RAG Pipeline (Week 2)
- [ ] Parse documents with Docling
- [ ] Implement document-type aware chunking
  - [ ] PDF documents: 1024 tokens, 100 overlap
  - [ ] ASPX web pages: 768 tokens, 80 overlap
  - [ ] Tables: 512 tokens, 0 overlap
  - [ ] Lists: 256 tokens, 50 overlap
- [ ] Set up Milvus vector database (3072-dim)
- [ ] Build multi-strategy retrieval pipeline
  - [ ] Semantic search (dense vectors)
  - [ ] Keyword search (BM25)
  - [ ] Hybrid search (weighted combination)
  - [ ] Cross-encoder re-ranking
- [ ] Implement grounded answer generation (Qwen 2.5 72B)
- [ ] Beat baseline metrics (+10% relevance)

### Stage 3: Multi-Agent System & API (Week 3)
- [ ] Implement 5-agent architecture
  - [ ] Agent 1: Query Understanding & Decomposition
  - [ ] Agent 2: Multi-Strategy Retrieval
  - [ ] Agent 3: Context Ranking & Filtering
  - [ ] Agent 4: Answer Generation & Citation
  - [ ] Agent 5: Verification & Quality Assurance
- [ ] Add performance optimizations (caching, streaming, parallel retrieval)
- [ ] Build FastAPI endpoint
- [ ] Deploy production system

### Stage 4: Optimization (Ongoing)
- [ ] Continuous evaluation
- [ ] Parameter tuning
- [ ] Performance optimization
- [ ] Prepare for blind test

---

## 🛠️ Tech Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Document Processing** | [Docling](https://github.com/DS4SD/docling) | Parse ASPX + PDF with structure preservation |
| **Vector Database** | [Milvus](https://milvus.io/) | Scalable semantic search (3072-dim vectors) |
| **Agent Framework** | [LangChain](https://python.langchain.com/) | Multi-agent orchestration |
| **Evaluation** | [RAGAS](https://docs.ragas.io/) | RAG-specific metrics (faithfulness, relevance) |
| **API** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance async REST API |
| **Embeddings** | OpenAI text-embedding-3-large | Best multilingual support (Hebrew/English, 3072-dim) |
| **LLM (Primary)** | Qwen 2.5 72B (Nebius Token Factory) | Excellent multilingual, strong reasoning, cost-effective |
| **LLM (Fallback)** | GPT-4o | Backup and comparison |
| **Re-ranking** | cross-encoder/ms-marco-MiniLM-L-12-v2 | Improves retrieval precision |

---

## 📊 Evaluation Metrics

| Metric | Weight | Baseline (GPT-5) | Our Target | Improvement |
|--------|--------|------------------|------------|-------------|
| **Relevance** | 65% | 75% | **90%** | +15% |
| **Citation Accuracy** | 15% | 85% | **95%** | +10% |
| **Efficiency** | 10% | 90% | **85%** | -5% (acceptable) |
| **Conversational Quality** | 10% | 80% | **90%** | +10% |
| **Total Weighted Score** | 100% | **78.5%** | **90.25%** | **+11.75%** |
| **Bonus: Voice** | +5% | ❌ | Lower priority | Week 3 if time |
| **Bonus: UI** | +5% | ❌ | Not planned | Focus on core |

**Key Performance Indicators**:
- **Hallucination rate**: <5%
- **Cost per query**: <$0.05
- **Latency p50**: <1s
- **Latency p95**: <2s
- **Latency p99**: <3s
- **Answer relevance (RAGAS)**: >0.9
- **Context precision (RAGAS)**: >0.85
- **Faithfulness (RAGAS)**: >0.9

---

## 👥 Team

**Team Size**: 3-4 participants

**Recommended Roles**:
- **Data & Retrieval Lead**: Web scraping, document processing, vector database, multi-strategy retrieval, re-ranking
- **Agent & Generation Lead**: 5-agent architecture, prompt engineering, answer generation, citation handling, verification
- **Evaluation & API Lead**: Evaluation framework, baseline testing, FastAPI development, performance optimization (caching, streaming, async)
- **Voice Lead** (Optional, Week 3): Voice integration (Whisper STT, TTS), Hebrew voice recognition

---

## 📚 Documentation

- **[DESIGN.md](DESIGN.md)** - Comprehensive system design and architecture
- **[QUICK_START.md](QUICK_START.md)** - Quick reference guide and daily checklists
- **[API Documentation](http://localhost:8000/docs)** - Interactive API docs (when server is running)

---

## 🔧 Development Workflow

### Running Tests

```bash
# Run all tests
pytest tests/

# Run specific test suite
pytest tests/test_retrieval.py

# Run with coverage
pytest --cov=src tests/
```

### Building the Vector Index

```bash
# Process all documents and build Milvus index
python scripts/build_index.py --data-dir data/raw --output-dir data/processed

# Build for specific domains
python scripts/build_index.py --domains car,life,travel
```

### Running Evaluation

```bash
# Evaluate on dev set
python scripts/run_evaluation.py --model rag --dataset data/evaluation/dev_set.json

# Compare multiple models
python scripts/run_evaluation.py --models gpt-4o,gpt-5.2,rag --dataset data/evaluation/reference_questions.json
```

### API Usage Examples

```python
import requests

# Ask a question
response = requests.post(
    "http://localhost:8000/chat",
    json={
        "question": "What does car insurance cover?",
        "domain": "car"  # Optional
    }
)

print(response.json())
# {
#   "answer": "Car insurance covers...",
#   "citations": [
#     {"source": "https://...", "section": "Coverage", "page": 3}
#   ],
#   "confidence": 0.95,
#   "domains": ["car"],
#   "conversation_id": "abc123"
# }
```

---

## 🎯 Milestones

### Week 1 (Feb 1-7): Foundation ✅
- [x] Data scraped and organized
- [x] Evaluation framework working
- [x] Baseline report completed

### Week 2 (Feb 8-14): RAG Core 🔄
- [ ] Documents processed and chunked
- [ ] Milvus index built
- [ ] RAG system beats baseline
- [ ] <5% hallucination rate achieved

### Week 3 (Feb 15-21): Production System ⏳
- [ ] Multi-agent system working
- [ ] FastAPI deployed
- [ ] Full evaluation on dev set
- [ ] System optimized for blind test

### Final (Feb 22): Presentation 🎤
- [ ] Demo ready
- [ ] Metrics documented
- [ ] Architecture explained
- [ ] Lessons learned prepared

---

## 🚨 Troubleshooting

### Milvus Connection Issues
```bash
# Check if Milvus is running
docker ps | grep milvus

# Restart Milvus
docker-compose restart milvus

# Check logs
docker-compose logs milvus
```

### Embedding Generation Fails
```bash
# Check OpenAI API key
echo $OPENAI_API_KEY

# Test API connection
python -c "from openai import OpenAI; client = OpenAI(); print(client.models.list())"
```

### Hebrew Text Encoding Issues
```python
# Ensure UTF-8 encoding
import sys
print(sys.getdefaultencoding())  # Should be 'utf-8'

# Set encoding explicitly
export PYTHONIOENCODING=utf-8
```

---

## 📈 Performance Benchmarks

### Target Performance (Week 3)

| Metric | Target | Current | Status |
|--------|--------|---------|--------|
| Answer Relevance | >85% | TBD | ⏳ |
| Citation Accuracy | >90% | TBD | ⏳ |
| Hallucination Rate | <5% | TBD | ⏳ |
| Latency (p50) | <1s | TBD | ⏳ |
| Latency (p95) | <2s | TBD | ⏳ |
| Cost per Query | <$0.05 | TBD | ⏳ |

---

## 🔐 Security & Privacy

- **API Keys**: Never commit `.env` file to version control
- **Data Privacy**: Ensure compliance with insurance data regulations
- **Rate Limiting**: Implement rate limiting on API endpoints
- **Input Validation**: Sanitize all user inputs
- **Logging**: Avoid logging sensitive customer information

---

## 🤝 Contributing

### Code Style
- Follow PEP 8 for Python code
- Use type hints for all functions
- Write docstrings for all public methods
- Keep functions focused and small (<50 lines)

### Git Workflow
```bash
# Create feature branch
git checkout -b feature/agent-routing

# Make changes and commit
git add .
git commit -m "feat: implement router agent with domain classification"

# Push and create PR
git push origin feature/agent-routing
```

### Commit Message Format
```
<type>: <description>

Types: feat, fix, docs, test, refactor, perf, chore
```

---

## 📞 Support & Resources

### Project Resources
- **Design Document**: [DESIGN.md](DESIGN.md)
- **Quick Start Guide**: [QUICK_START.md](QUICK_START.md)
- **Task List**: Run `python scripts/view_tasks.py`

### External Resources
- [Docling Documentation](https://github.com/DS4SD/docling)
- [Milvus Documentation](https://milvus.io/docs)
- [LangChain Documentation](https://python.langchain.com/)
- [RAGAS Documentation](https://docs.ragas.io/)
- [FastAPI Documentation](https://fastapi.tiangolo.com/)

### Research Papers
- [RAG Paper (Lewis et al., 2020)](https://arxiv.org/abs/2005.11401)
- [Self-RAG (Asai et al., 2023)](https://arxiv.org/abs/2310.11511)
- [Lost in the Middle (Liu et al., 2023)](https://arxiv.org/abs/2307.03172)

---

## 📝 License

MIT License - See [LICENSE](LICENSE) file for details

---

## 🙏 Acknowledgments

- **Harel Insurance** for providing the domain and data
- **Docling Team** for document processing tools
- **Milvus Community** for vector database support
- **LangChain Team** for agent framework
- **RAGAS Team** for evaluation metrics

---

## 📊 Project Status

**Current Phase**: Planning & Design ✅
**Next Phase**: Project Setup & Data Collection
**Timeline**: On track for Feb 22 presentation
**Team Status**: Ready to begin implementation

---

**Built with ❤️ for Harel Insurance | February 2026**


