# Quick Start Guide - Harel Insurance Chatbot

## 🎯 Project at a Glance

**What**: Production-grade customer support chatbot for Harel Insurance  
**When**: February 1-21, 2026 (3 weeks)  
**Goal**: Beat GPT-5 baseline using RAG + Agentic design  
**Domains**: 8 insurance types (Car, Life, Travel, Health, Dental, Mortgage, Business, Apartment)

---

## 📅 3-Week Timeline

### Week 1 (Feb 1-7): Foundation
**Focus**: Baseline & Evaluation
- Scrape Harel insurance data (~350 documents)
- Build evaluation framework (RAGAS + custom metrics)
- Run GPT-4o and GPT-5.2 baselines
- **Deliverable**: Baseline report with failure analysis

### Week 2 (Feb 8-14): RAG Core
**Focus**: Retrieval-Augmented Generation
- Parse documents with Docling (ASPX + PDF)
- Implement intelligent chunking (512 tokens, structure-aware)
- Set up Milvus vector database
- Build grounded answer generation with citations
- **Deliverable**: RAG system beating baseline (<5% hallucination)

### Week 3 (Feb 15-21): Production System
**Focus**: Agents & API
- Design multi-agent architecture (router + 8 domain agents)
- Implement conversation management
- Deploy FastAPI endpoint
- Optimize performance (latency, cost)
- **Deliverable**: Production-ready API

---

## 🏗️ Architecture Overview

```
User Query
    ↓
Agent 1: Query Understanding & Decomposition
    ↓
Agent 2: Multi-Strategy Retrieval (Semantic + Keyword + Hybrid)
    ↓
Agent 3: Context Ranking & Filtering (Cross-Encoder Re-ranking)
    ↓
Agent 4: Answer Generation & Citation (Qwen 2.5 72B)
    ↓
Agent 5: Verification & Quality Assurance
    ↓
Grounded Answer + Citations
```

**Key Innovation**: 5-agent pipeline with multi-stage verification beats simple RAG

---

## 🛠️ Tech Stack

| Component | Tool | Why |
|-----------|------|-----|
| **Doc Processing** | Docling | Handles ASPX + PDF, preserves structure |
| **Vector DB** | Milvus | Production-ready, scalable |
| **Agents** | LangChain | Rich ecosystem, easy orchestration |
| **Evaluation** | RAGAS | RAG-specific metrics |
| **API** | FastAPI | High performance, async |
| **Embeddings** | text-embedding-3-large | Best multilingual (Hebrew/English) |
| **LLM** | Qwen 2.5 72B (Nebius) | Excellent multilingual, cost-effective |
| **Re-ranking** | cross-encoder | Improves retrieval precision |

---

## 📊 Success Metrics

| Metric | Weight | Target |
|--------|--------|--------|
| **Relevance** | 65% | +15% vs GPT-5 |
| **Citation Accuracy** | 15% | >90% |
| **Efficiency** | 10% | <2s latency |
| **Conversational Quality** | 10% | High clarity |
| **Bonus: Voice** | +5% | Optional |
| **Bonus: UI** | +5% | Optional |

**Critical Targets**:
- Hallucination rate: <5%
- Cost per query: <$0.05

---

## 🚀 Getting Started (First Day)

### 1. Environment Setup (30 min)
```bash
# Clone and setup
git clone <repo-url>
cd harel-insurance-chatbot
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Configure API keys
cp .env.example .env
# Add: OPENAI_API_KEY, MILVUS_HOST, etc.
```

### 2. Start Milvus (10 min)
```bash
docker-compose up -d milvus
curl http://localhost:19530/healthz  # Verify
```

### 3. Scrape Initial Data (2 hours)
```bash
python scripts/run_scraping.py --domains car,life,travel
```

### 4. Run Baseline Test (1 hour)
```bash
python scripts/run_baseline.py --model gpt-4o --questions data/evaluation/reference_questions.json
```

---

## 🎯 Key Design Decisions

### 1. Agent Architecture
**Decision**: 5-agent role-based system
- Agent 1: Query Understanding & Decomposition
- Agent 2: Multi-Strategy Retrieval
- Agent 3: Context Ranking & Filtering
- Agent 4: Answer Generation & Citation
- Agent 5: Verification & Quality Assurance

### 2. LLM Choice
**Decision**: Qwen 2.5 72B via Nebius Token Factory
- Excellent multilingual support (Hebrew/English)
- Strong reasoning capabilities
- Cost-effective
- Model-agnostic design allows easy swapping

### 3. Chunking Strategy
**Decision**: Document-type aware chunking
- PDF documents: 1024 tokens, 100 overlap
- ASPX web pages: 768 tokens, 80 overlap
- Tables: 512 tokens, 0 overlap
- Lists: 256 tokens, 50 overlap

### 4. Embeddings
**Decision**: OpenAI text-embedding-3-large
- Best multilingual support for Hebrew
- 3072 dimensions
- Proven performance

### 5. Voice/UI Bonuses
**Decision**: Voice support lower priority
- Week 1-2: Core RAG system
- Week 3: Add voice if ahead of schedule
- Voice worth +5%

---

## 🚨 Common Pitfalls to Avoid

1. **Poor Chunking** → Broken context, bad citations
   - Solution: Test chunking early, preserve structure

2. **Hallucinations** → Critical failure
   - Solution: Strict prompts, citation validation, confidence thresholds

3. **Hebrew Text Issues** → Encoding problems, poor embeddings
   - Solution: Test multilingual embeddings early

4. **Scope Creep** → Running out of time
   - Solution: Focus on Stages 1-2, simplify Stage 3 if needed

5. **No Evaluation** → Can't measure progress
   - Solution: Build eval framework in Week 1, run daily

---

## 📝 Daily Checklist

### Week 1 Daily
- [ ] Scrape data for 1-2 domains
- [ ] Add evaluation metrics
- [ ] Test baseline on new questions
- [ ] Document failure patterns

### Week 2 Daily
- [ ] Process documents for 1-2 domains
- [ ] Test chunking strategies
- [ ] Run RAG evaluation
- [ ] Iterate on prompts

### Week 3 Daily
- [ ] Implement 1-2 agents
- [ ] Test cross-domain queries
- [ ] Optimize latency/cost
- [ ] Prepare demo

---

## 🤝 Team Roles (3-4 people)

**Role 1: Data & Retrieval**
- Scraping, parsing, chunking, Milvus setup

**Role 2: Agents & Generation**
- Agent architecture, prompts, answer generation

**Role 3: Evaluation & API**
- Metrics, baseline testing, FastAPI, optimization

**Role 4 (Optional): UI/Voice**
- Web interface, voice integration

---

## 📚 Essential Reading

1. **DESIGN.md** - Full system design (read first!)
2. **README.md** - Project overview
3. [RAGAS Docs](https://docs.ragas.io/) - Evaluation metrics
4. [Docling Docs](https://github.com/DS4SD/docling) - Document parsing
5. [Milvus Docs](https://milvus.io/docs) - Vector database

---

## 🎬 Next Steps

1. ✅ Read DESIGN.md thoroughly
2. ✅ Set up development environment
3. ✅ Assign team roles
4. ⏭️ Start scraping Harel data
5. ⏭️ Build evaluation framework

**Questions?** Review DESIGN.md Section: "Questions to Discuss"

---

**Ready to build? Let's go! 🚀**

