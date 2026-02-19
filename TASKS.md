# Harel Insurance Chatbot - Task List

**Last Updated:** 2026-02-19  
**Current Phase:** Phase 2 - RAG Implementation

---

## ✅ Completed

### Phase 1: Baseline Evaluation
- [x] Save evaluation dataset (20 Hebrew questions, 5 domains)
- [x] Create evaluation harness with RAGAS metrics
- [x] Implement baseline runner (supports Nebius Token Factory)
- [x] Run baseline evaluation
- [x] Generate baseline report

**Results:**
| Model | Strategy | Accuracy | Latency |
|-------|----------|----------|---------|
| Qwen3-235B | basic | 50% | 5.8s |
| Qwen3-235B | strict | **55%** | 0.7s |
| Llama-3.3-70B | basic | 40% | 22.4s |
| DeepSeek-V3 | strict | 45% | 1.5s |

---

## 🔄 In Progress

### Phase 2: RAG Implementation
**Goal:** Improve accuracy from 55% → 85%+

#### 2.1 Document Ingestion Pipeline [IN PROGRESS]
- [x] Create `src/ingestion/__init__.py`
- [x] **Document Registry** - Track indexed files for incremental updates
- [x] **Topic Taxonomy** - Define topic hierarchy (9 domains + subtopics)
- [x] **PDF Processor** - Extract text with Docling (page-by-page)
- [ ] **Topic Classifier** - Classify chunks into topics
- [ ] **Document Indexer** - Orchestrate full pipeline
- [ ] **Ingestion CLI** - `scripts/ingest_documents.py`

#### 2.2 Vector Store Setup [NOT STARTED]
- [ ] Set up embeddings (text-embedding-3-large, 3072-dim)
- [ ] Configure Milvus/ChromaDB
- [ ] Create collection schema
- [ ] Implement batch embedding

#### 2.3 Retrieval Pipeline [NOT STARTED]
- [ ] Semantic search (dense vectors)
- [ ] BM25 search (sparse, keyword)
- [ ] Hybrid search (weighted combination)
- [ ] Query expansion for Hebrew

#### 2.4 Context Ranking [NOT STARTED]
- [ ] Cross-encoder re-ranking (ms-marco-MiniLM-L-12-v2)
- [ ] Top-K selection

#### 2.5 Answer Generation with Citations [NOT STARTED]
- [ ] RAG prompt template
- [ ] Citation format: `[מקור: filename, עמוד X]`
- [ ] "I don't know" handling

#### 2.6 RAG Evaluation [NOT STARTED]
- [ ] Re-run evaluation with RAG
- [ ] Compare vs baseline
- [ ] Generate improvement report

---

## 📋 Backlog

### Stage 3: Agentic System & API
- [ ] Design multi-agent architecture
- [ ] Implement routing logic
- [ ] Build domain-specific agents
- [ ] Conversation management
- [ ] FastAPI endpoint
- [ ] Error handling & logging

### Stage 4: Optimization
- [ ] Continuous evaluation
- [ ] Retrieval parameter tuning
- [ ] Prompt optimization
- [ ] Latency & cost optimization
- [ ] Prepare for blind test

### Optional (Bonus)
- [ ] Voice interface (+5%)
- [ ] Web UI (+5%)

---

## 📊 Key Metrics

| Metric | Baseline | Target | Current |
|--------|----------|--------|---------|
| Accuracy | 55% | 85%+ | - |
| Citation Rate | 0% | 95%+ | - |
| Hallucination Rate | ~25% | <5% | - |
| Latency (p95) | 5.8s | <3s | - |

---

## 📁 Key Files

| File | Purpose |
|------|---------|
| `data/evaluation/dev_set.json` | 20 Hebrew evaluation questions |
| `data/evaluation/results/BASELINE_REPORT.md` | Phase 1 results |
| `scripts/run_quick_baseline.py` | Baseline evaluation script |
| `src/processing/chunker.py` | Contextual chunking (no overlap) |
| `config/config.yaml` | System configuration |

---

## 🔧 Infrastructure

- **LLM Provider:** Nebius Token Factory
- **Primary Model:** Qwen3-235B-A22B-Instruct-2507
- **Embeddings:** text-embedding-3-large (3072-dim)
- **Vector DB:** Milvus (configured in docker-compose)
- **Evaluation:** RAGAS 0.4.x

