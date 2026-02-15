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

```
User Query → Router Agent → Domain Agent(s) → RAG Pipeline → Grounded Answer
                ↓                                    ↓
         Classification                    Milvus Vector Search
                                                     ↓
                                          Context + Citations
```

**Key Components**:
1. **Multi-Agent System**: Router + 8 domain-specific agents + synthesis agent
2. **RAG Pipeline**: Docling parsing → Milvus retrieval → LLM generation
3. **Evaluation Framework**: RAGAS + custom citation validator
4. **FastAPI**: Production-ready REST API

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
- [ ] Implement intelligent chunking
- [ ] Set up Milvus vector database
- [ ] Build retrieval pipeline
- [ ] Implement grounded answer generation
- [ ] Beat baseline metrics

### Stage 3: Agentic System & API (Week 3)
- [ ] Design multi-agent architecture
- [ ] Implement router and domain agents
- [ ] Build FastAPI endpoint
- [ ] Add conversation management
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
| **Vector Database** | [Milvus](https://milvus.io/) | Scalable semantic search |
| **Agent Framework** | [LangChain](https://python.langchain.com/) | Multi-agent orchestration |
| **Evaluation** | [RAGAS](https://docs.ragas.io/) + [Opik](https://www.comet.com/site/products/opik/) | RAG metrics + observability |
| **API** | [FastAPI](https://fastapi.tiangolo.com/) | High-performance REST API |
| **Embeddings** | OpenAI text-embedding-3-large | Multilingual embeddings (Hebrew/English) |
| **LLM** | GPT-4o / Llama 3.1 / Mixtral | Answer generation |

---

## 📊 Evaluation Metrics

| Metric | Weight | Target | Current |
|--------|--------|--------|---------|
| **Relevance** | 65% | +15% vs baseline | TBD |
| **Citation Accuracy** | 15% | >90% | TBD |
| **Efficiency** | 10% | <2s latency | TBD |
| **Conversational Quality** | 10% | High clarity | TBD |
| **Bonus: Voice** | +5% | Implemented | ❌ |
| **Bonus: UI** | +5% | Implemented | ❌ |

**Key Targets**:
- Hallucination rate: <5%
- Cost per query: <$0.05
- Latency (p95): <2 seconds

---

## 👥 Team

**Team Size**: 3-4 participants

**Recommended Roles**:
- **Data & Retrieval Lead**: Web scraping, document processing, vector database, retrieval optimization
- **Agent & Generation Lead**: Agent architecture, prompt engineering, answer generation, citation handling
- **Evaluation & API Lead**: Evaluation framework, baseline testing, FastAPI development, performance optimization
- **UI/Voice Lead** (Optional): Web interface, voice integration, user experience

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


