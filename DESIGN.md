# Harel Insurance Chatbot - System Design Document

**Project**: Domain-Specific Customer Support Chatbot for Harel Insurance
**Timeline**: February 1-21, 2026 (3 weeks)
**Presentation**: February 22, 2026
**Team Size**: 3-4 participants

---

## Table of Contents
1. [System Architecture](#system-architecture)
2. [Multi-Agent Architecture](#multi-agent-architecture)
3. [Technology Stack](#technology-stack)
4. [Performance Optimization](#performance-optimization)
5. [Document Processing Strategy](#document-processing-strategy)
6. [Implementation Stages](#implementation-stages)
7. [Success Metrics](#success-metrics)
8. [Risk Mitigation](#risk-mitigation)
9. [Milestones & Checkpoints](#milestones--checkpoints)
10. [Next Steps](#next-steps)

---

## 🎯 Project Overview

**Goal**: Build a production-grade, domain-specific customer support chatbot for Harel Insurance that outperforms GPT-5 baseline using retrieval-augmented generation (RAG) and multi-agent system design.

**Success Criteria**:
- Beat GPT-5.2 baseline on relevance, citation accuracy, and hallucination reduction
- Handle 8 insurance domains: Car, Life, Travel, Health, Dental, Mortgage, Business, Apartment
- Provide grounded answers with explicit citations from source documents
- Production-ready API deployment with <2s latency

**Evaluation Metrics**:

| Metric | Weight | Target |
|--------|--------|--------|
| **Relevance** | 65% | +15% vs GPT-5 baseline |
| **Citation Accuracy** | 15% | >90% accuracy |
| **Efficiency** | 10% | <2s p95 latency, <$0.05/query |
| **Conversational Quality** | 10% | High clarity and professionalism |
| **Bonus: Voice Support** | +5% | Lower priority (Week 3 if time) |
| **Bonus: UI Polish** | +5% | Optional |

---

## System Architecture

### High-Level Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      User Interface Layer                        │
│  - FastAPI REST Endpoint (Primary)                               │
│  - Voice Interface (Optional +5% - Lower Priority)               │
│  - Web UI (Optional +5%)                                         │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                  5-Agent Orchestration Layer                     │
│                                                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Agent 1: Query Understanding & Decomposition               │ │
│  │ Skills: decompose_query, identify_intent, classify_type,   │ │
│  │         detect_domain_overlap, extract_entities            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Agent 2: Multi-Strategy Retrieval                          │ │
│  │ Skills: semantic_search, keyword_search, hybrid_search,    │ │
│  │         query_expansion, cross_domain_retrieval            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Agent 3: Context Ranking & Filtering                       │ │
│  │ Skills: rerank_by_relevance, deduplicate, filter_quality,  │ │
│  │         diversity_sampling, context_compression            │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Agent 4: Answer Generation & Citation                      │ │
│  │ Skills: generate_grounded_answer, attach_citations,        │ │
│  │         handle_multi_source, detect_conflicts              │ │
│  └────────────────────────────────────────────────────────────┘ │
│                              ↓                                   │
│  ┌────────────────────────────────────────────────────────────┐ │
│  │ Agent 5: Verification & Quality Assurance                  │ │
│  │ Skills: verify_citations, detect_hallucinations,           │ │
│  │         check_completeness, validate_consistency           │ │
│  └────────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                      RAG Pipeline Layer                          │
│                                                                   │
│  ┌──────────────────┐         ┌──────────────────────────────┐ │
│  │ Retrieval Engine │    →    │  Answer Generation           │ │
│  │                  │         │                              │ │
│  │ - Milvus Vector  │         │  - LLM (Qwen 2.5 72B)       │ │
│  │   Search         │         │  - Model-Agnostic Interface  │ │
│  │ - Hybrid Search  │         │  - Grounded Generation       │ │
│  │   (Dense+Sparse) │         │  - Citation Attachment       │ │
│  │ - Cross-Encoder  │         │  - Hallucination Detection   │ │
│  │   Re-ranking     │         │                              │ │
│  └──────────────────┘         └──────────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│                    Data & Storage Layer                          │
│                                                                   │
│  ┌──────────────┐  ┌──────────────┐  ┌────────────────────┐    │
│  │ Milvus       │  │ Document     │  │ Metadata DB        │    │
│  │ Vector DB    │  │ Storage      │  │ (Citations,        │    │
│  │              │  │ (Parsed      │  │  Sources)          │    │
│  │ - 3072-dim   │  │  Chunks)     │  │                    │    │
│  │   embeddings │  │              │  │                    │    │
│  │ - IVF_FLAT   │  │              │  │                    │    │
│  │   index      │  │              │  │                    │    │
│  └──────────────┘  └──────────────┘  └────────────────────┘    │
└─────────────────────────────────────────────────────────────────┘
                              ↑
┌─────────────────────────────────────────────────────────────────┐
│                  Data Processing Pipeline                        │
│                                                                   │
│  Web Scraper → Docling Parser → Intelligent Chunker → Embedder  │
│  (Harel site)  (ASPX + PDF)     (Doc-type aware)    (OpenAI)    │
│                                  (1024 base size)                │
└─────────────────────────────────────────────────────────────────┘
```

---



## Multi-Agent Architecture

### 5-Agent Competitive Architecture

**Design Philosophy**: Role-based agents with specialized skills to maximize performance and beat rival teams.

**Why 5 Agents?**
- **Separation of Concerns**: Each agent has a focused responsibility
- **Optimization**: Can optimize each stage independently
- **Quality Assurance**: Multi-stage verification catches errors
- **Competitive Edge**: Most teams will use simple RAG; we use sophisticated pipeline

---

### Agent 1: Query Understanding & Decomposition 🧠

**Role**: Break down complex queries and understand user intent

**Skills**:
1. `decompose_complex_query` - Split "What's covered in car and life insurance?" into sub-queries
2. `identify_implicit_intent` - Detect hidden needs (e.g., "I'm traveling" → travel insurance)
3. `classify_question_type` - Coverage? Claim? Premium? Eligibility?
4. `detect_domain_overlap` - Identify cross-domain questions
5. `extract_entities` - Pull out: policy types, coverage amounts, dates, etc.

**Example Input/Output**:
```python
Input: "If I have an accident abroad, am I covered by car or travel insurance?"

Output: {
    "sub_queries": [
        "What does car insurance cover for accidents abroad?",
        "What does travel insurance cover for accidents abroad?"
    ],
    "question_type": "coverage_comparison",
    "domains": ["car", "travel"],
    "entities": ["accident", "abroad"],
    "complexity": "high",
    "requires_synthesis": True
}
```

**Competitive Advantage**:
- ✅ Handles complex questions better than single-shot retrieval
- ✅ Improves relevance by understanding nuance (65% of score!)
- ✅ Enables parallel retrieval for sub-queries

---

### Agent 2: Multi-Strategy Retrieval 🔍

**Role**: Retrieve relevant documents using multiple strategies in parallel

**Skills**:
1. `semantic_search` - Dense vector retrieval (Milvus)
2. `keyword_search` - BM25 sparse retrieval (for exact terms)
3. `hybrid_search` - Combine semantic + keyword (weighted)
4. `metadata_filtering` - Filter by domain, document type, date
5. `query_expansion` - Generate alternative phrasings
6. `cross_domain_retrieval` - Retrieve from multiple domains in parallel

**Competitive Advantage**:
- ✅ Hybrid search beats pure vector search (proven in research)
- ✅ Query expansion catches synonyms and variations
- ✅ Sub-query retrieval handles complex questions
- ✅ Higher recall → better relevance

---

### Agent 3: Context Ranking & Filtering 📊

**Role**: Re-rank and filter retrieved documents for quality

**Skills**:
1. `rerank_by_relevance` - Use cross-encoder for precise ranking
2. `deduplicate_context` - Remove redundant information
3. `filter_low_quality` - Remove irrelevant or contradictory docs
4. `diversity_sampling` - Ensure diverse perspectives (MMR)
5. `context_compression` - Summarize long documents
6. `citation_preparation` - Extract citation metadata

**Competitive Advantage**:
- ✅ Cross-encoder re-ranking significantly improves relevance
- ✅ Deduplication reduces noise
- ✅ Diversity ensures comprehensive answers
- ✅ Better context → better answers

---

### Agent 4: Answer Generation & Citation ✍️

**Role**: Generate grounded answers with precise inline citations

**Skills**:
1. `generate_grounded_answer` - Answer strictly from context
2. `attach_inline_citations` - Cite every claim with [1], [2], etc.
3. `handle_multi_source` - Synthesize from multiple documents
4. `detect_conflicts` - Identify contradictory information
5. `format_answer` - Structure for clarity
6. `generate_confidence_score` - Self-assess answer quality

**Competitive Advantage**:
- ✅ Inline citations improve citation accuracy (15% of score)
- ✅ Conflict detection shows thoroughness
- ✅ Confidence scoring enables fallback handling
- ✅ Strict grounding reduces hallucinations

---

### Agent 5: Verification & Quality Assurance ✅

**Role**: Validate answer quality and catch errors before submission

**Skills**:
1. `verify_citations` - Check every citation is valid
2. `detect_hallucinations` - Compare answer to context
3. `check_completeness` - Ensure question is fully answered
4. `validate_consistency` - Check for contradictions
5. `assess_conversational_quality` - Evaluate clarity and tone
6. `trigger_fallback` - Reject low-quality answers

**Verification Process**:
```python
async def verify_answer(self, query, answer, context):
    issues = []

    # Check 1: Citation validation
    if not self.verify_all_citations(answer, context)["valid"]:
        issues.append("Invalid citations detected")

    # Check 2: Hallucination detection
    hallucination_check = await self.detect_hallucinations(answer, context)
    if hallucination_check["hallucination_score"] > 0.1:
        issues.append(f"Potential hallucination: {hallucination_check['claims']}")

    # Check 3: Completeness
    if await self.check_completeness(query, answer) < 0.8:
        issues.append("Answer may be incomplete")

    # Decision: Accept or reject
    if len(issues) == 0:
        return {"status": "approved", "answer": answer}
    elif len(issues) <= 2 and "hallucination" not in str(issues):
        return {"status": "approved_with_warnings", "answer": answer}
    else:
        return {"status": "rejected", "fallback": self.generate_fallback()}
```

**Competitive Advantage**:
- ✅ Catches errors before they hurt our score
- ✅ Ensures citation accuracy (15% of score)
- ✅ Reduces hallucinations (critical failure)
- ✅ Improves conversational quality (10% of score)

---

### Complete Workflow

```python
async def process_query(self, user_query: str):
    # Agent 1: Understand and decompose
    analysis = await self.agent1_query_understanding.analyze(user_query)

    # Agent 2: Multi-strategy retrieval (parallel)
    retrieved_docs = await self.agent2_retrieval.retrieve(analysis)

    # Agent 3: Rank and filter
    ranked_context = await self.agent3_ranking.rank_and_filter(user_query, retrieved_docs)

    # Agent 4: Generate answer with citations
    answer_result = await self.agent4_generation.generate_answer(user_query, ranked_context)

    # Agent 5: Verify quality
    verification = await self.agent5_verification.verify_answer(
        user_query, answer_result["answer"], ranked_context
    )

    # Return or fallback
    return verification["answer"] if verification["status"] == "approved" else verification["fallback"]
```

---

## Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| **Document Processing** | Docling | Handles ASPX + PDF, preserves structure (tables, lists) |
| **Vector Database** | Milvus | Production-ready, scalable, supports hybrid search |
| **Agent Framework** | LangChain | Rich ecosystem, agent orchestration, tool integration |
| **Evaluation** | RAGAS | RAG-specific metrics (faithfulness, relevance, precision) |
| **API Framework** | FastAPI | High performance, async support, auto-documentation |
| **Embeddings** | OpenAI text-embedding-3-large | Best multilingual support (Hebrew/English), 3072-dim |
| **LLM (Primary)** | Qwen 2.5 72B (Nebius Token Factory) | Excellent multilingual, strong reasoning, cost-effective |
| **LLM (Baseline)** | GPT-4o / GPT-5.2 | For comparison and fallback |
| **Re-ranking** | cross-encoder/ms-marco-MiniLM-L-12-v2 | Improves retrieval precision |
| **Deployment** | Docker + Docker Compose | Easy setup, reproducible environment |

### Model-Agnostic LLM Interface

**Design**: Abstract LLM provider to easily swap models

```python
class LLMProvider(ABC):
    @abstractmethod
    async def generate(self, prompt: str, **kwargs) -> str:
        pass

    @abstractmethod
    async def astream(self, prompt: str, **kwargs) -> AsyncIterator[str]:
        pass

# Implementations
class QwenProvider(LLMProvider):
    # Qwen 2.5 72B via Nebius Token Factory
    ...

class OpenAIProvider(LLMProvider):
    # GPT-4o / GPT-5.2
    ...

class ClaudeProvider(LLMProvider):
    # Claude 3.5 Sonnet (for comparison)
    ...

# Easy model switching via config
llm = LLMFactory.create(
    provider=os.getenv("LLM_PROVIDER", "qwen"),
    api_key=os.getenv("LLM_API_KEY"),
    base_url=os.getenv("LLM_BASE_URL")
)
```

**Benefits**:
- ✅ Swap models with 1 config change
- ✅ Easy A/B testing
- ✅ Can run multiple models in parallel for comparison
- ✅ Future-proof

---

## Performance Optimization

### Comprehensive Latency Optimization Strategy

**Target Latency**:
- p50: <1s
- p95: <2s
- p99: <3s

### Optimization Techniques

#### 1. Parallel Retrieval
```python
# Retrieve from multiple domains simultaneously
async def parallel_retrieve(self, domains, query):
    tasks = [self.retrieve(domain, query) for domain in domains]
    results = await asyncio.gather(*tasks)
    return results
```
**Impact**: 50-70% latency reduction for multi-domain queries

#### 2. Streaming Responses
```python
# Stream answer as it's generated
async def stream_response(self, query):
    async for chunk in self.llm.astream(prompt):
        yield chunk  # User sees response immediately
```
**Impact**: 80% perceived latency reduction

#### 3. Caching
```python
# Cache embeddings for common queries
@lru_cache(maxsize=1000)
def cache_embeddings(self, text):
    return self.embed(text)

# Cache complete answers for frequent questions
def cache_answers(self, query):
    cache_key = self.normalize_query(query)
    if cache_key in self.answer_cache:
        return self.answer_cache[cache_key]
```
**Impact**: 90% latency reduction for cache hits

#### 4. Speculative Execution
```python
# Start retrieval before classification completes
async def speculative_retrieve(self, query):
    # Quick keyword-based guess (50ms)
    likely_domain = self.quick_classify(query)

    # Start retrieval immediately
    retrieval_task = asyncio.create_task(self.retrieve(likely_domain, query))

    # Confirm with LLM in parallel
    confirmed_analysis = await self.analyst_agent.analyze(query)

    # If guess was right, we saved time!
    if confirmed_analysis["domains"][0] == likely_domain:
        return await retrieval_task
    else:
        retrieval_task.cancel()
        return await self.retrieve(confirmed_analysis["domains"][0], query)
```
**Impact**: 30-40% latency reduction

#### 5. Batch Processing
```python
# Embed multiple texts in one API call
async def batch_embed(self, texts):
    return await self.embedder.embed_batch(texts, batch_size=100)
```
**Impact**: 20-30% latency reduction

#### 6. Connection Pooling
```python
def __init__(self):
    self.milvus_pool = ConnectionPool(max_connections=10)
    self.http_session = aiohttp.ClientSession()  # Reuse connections
```
**Impact**: 10-15% latency reduction

#### 7. Prefetching
```python
# Preload frequently accessed documents into memory
async def prefetch_common_docs(self):
    common_queries = ["car insurance coverage", "life insurance premium"]
    for query in common_queries:
        await self.retrieve(query)  # Warms up cache
```
**Impact**: 50% latency reduction for common queries

#### 8. Context Compression
```python
# Compress retrieved context to reduce tokens
def compress_context(self, docs):
    # Remove redundant information
    # Summarize if too long
    return compressed_docs
```
**Impact**: 15-25% latency reduction

#### 9. Early Termination
```python
# Stop retrieving once we have high-confidence match
async def retrieve_with_threshold(self, query, threshold=0.8):
    async for doc in self.stream_retrieve(query):
        if doc.score > threshold:
            return [doc]  # Good enough, stop searching
```
**Impact**: 20-40% latency reduction

#### 10. Async Everything
```python
# Fully async pipeline
async def process_query(self, query):
    analysis = await self.analyst_agent.analyze(query)
    docs = await self.retrieval_agent.retrieve(analysis)
    response = await self.response_builder.build(query, docs)
    return response
```
**Impact**: 30-50% latency reduction

### Implementation Priority

**Week 1** (Quick wins):
- ✅ Async pipeline
- ✅ Connection pooling
- ✅ Batch processing

**Week 2** (Medium effort):
- ✅ Caching (embeddings + answers)
- ✅ Parallel retrieval
- ✅ Streaming responses

**Week 3** (Advanced):
- ✅ Speculative execution
- ✅ Prefetching
- ✅ Compression
- ✅ Early termination

---


## Document Processing Strategy

### Document-Type Aware Chunking

**Key Decision**: Different document types need different chunking strategies

| Document Type | Chunk Size | Overlap | Strategy | Preserve |
|---------------|------------|---------|----------|----------|
| **PDF Documents** (formal policies) | 1024 tokens | 100 tokens | Semantic (respect paragraphs) | Tables, lists, clauses |
| **ASPX Web Pages** (FAQs, guides) | 768 tokens | 80 tokens | Section-based (H2/H3 headers) | Q&A pairs, bullet lists |
| **Tables** | 512 tokens | 0 tokens | Table-aware (keep entire table) | Headers, rows |
| **Lists** (coverage items) | 256 tokens | 50 tokens | List-aware (keep related items) | Parent context |

### Implementation

```python
def chunk_document(doc, doc_type):
    if doc_type == "pdf":
        return semantic_chunk(doc, size=1024, overlap=100)
    elif doc_type == "aspx":
        return section_chunk(doc, size=768, overlap=80)
    elif doc.has_tables:
        return table_aware_chunk(doc, size=512)
    else:
        return default_chunk(doc, size=1024)
```

### Rationale

**Why 1024 base chunk size?**
- Insurance policy clauses typically 200-800 tokens
- Fits most policy descriptions completely
- Better context preservation than 512
- Embedding model (text-embedding-3-large) supports up to 8191 tokens
- Reduces context fragmentation

**Why document-type specific?**
- PDFs: Formal, structured → larger chunks preserve legal language
- ASPX: Web content, shorter sections → medium chunks
- Tables: Structured data → keep complete for accuracy
- Lists: Related items → smaller chunks with overlap

### Hebrew Text Handling

**Challenges**:
- Right-to-left (RTL) text direction
- Mixed Hebrew/English content
- Nikud (vowel marks) normalization
- Encoding issues

**Solutions**:
```python
def preprocess_hebrew_text(text):
    # Normalize Unicode
    text = unicodedata.normalize('NFC', text)

    # Remove nikud (optional, improves matching)
    text = remove_nikud(text)

    # Handle mixed RTL/LTR
    text = normalize_bidirectional_text(text)

    return text
```

**Embedding Strategy**:
- Use OpenAI text-embedding-3-large (proven Hebrew support)
- Test early with Hebrew queries (Week 1)
- Fallback: BGE-M3 (open-source multilingual)

---

## Implementation Stages

### Stage 1: Foundation & Baseline (Week 1: Feb 1-7)

**Objective**: Establish baseline and evaluation framework

**Tasks**:
1. **Data Collection**
   - Scrape Harel insurance website (~350 documents)
   - Focus on 2-3 domains initially (Car, Life, Travel)
   - Store raw ASPX and PDF files
   - Document data structure and sources

2. **Evaluation Framework**
   - Set up RAGAS evaluation pipeline
   - Load reference questions (Reference_Questions.json)
   - Define custom metrics (citation accuracy, hallucination rate)
   - Create dev/test split

3. **Baseline Testing**
   - Run GPT-4o baseline on reference questions
   - Run GPT-5.2 baseline (if available)
   - Document failure patterns
   - Identify improvement opportunities

4. **Infrastructure Setup**
   - Deploy Milvus via Docker Compose
   - Set up development environment
   - Configure API keys (OpenAI, Nebius)
   - Create project structure

**Deliverables**:
- ✅ Raw data collected and organized
- ✅ Evaluation framework working
- ✅ Baseline report with metrics
- ✅ Infrastructure deployed

**Success Criteria**:
- At least 100 documents scraped
- Baseline evaluation runs successfully
- Clear understanding of GPT-5 performance

---

### Stage 2: RAG Pipeline (Week 2: Feb 8-14)

**Objective**: Build core RAG system that beats baseline

**Tasks**:
1. **Document Processing**
   - Parse documents with Docling
   - Implement document-type aware chunking
   - Test Hebrew text handling
   - Generate embeddings (OpenAI text-embedding-3-large)

2. **Vector Database**
   - Create Milvus collection (3072-dim)
   - Index all processed documents
   - Implement metadata filtering
   - Test retrieval quality

3. **Retrieval Pipeline**
   - Implement semantic search
   - Add keyword search (BM25)
   - Build hybrid search
   - Add cross-encoder re-ranking

4. **Answer Generation**
   - Deploy Qwen 2.5 72B on Nebius
   - Implement grounded generation prompts
   - Add citation attachment
   - Test on dev set

**Deliverables**:
- ✅ All documents processed and indexed
- ✅ Retrieval pipeline working
- ✅ RAG system beats baseline
- ✅ <5% hallucination rate

**Success Criteria**:
- Relevance: +10% vs baseline
- Citation accuracy: >85%
- Hallucination rate: <5%

---

### Stage 3: Multi-Agent System (Week 3: Feb 15-21)

**Objective**: Implement 5-agent architecture and optimize

**Tasks**:
1. **Agent Implementation**
   - Agent 1: Query Understanding & Decomposition
   - Agent 2: Multi-Strategy Retrieval
   - Agent 3: Context Ranking & Filtering
   - Agent 4: Answer Generation & Citation
   - Agent 5: Verification & Quality Assurance

2. **Performance Optimization**
   - Implement async pipeline
   - Add caching (embeddings + answers)
   - Enable parallel retrieval
   - Add streaming responses
   - Implement speculative execution

3. **API Development**
   - Build FastAPI endpoints (/chat, /health, /metrics)
   - Add request validation
   - Implement rate limiting
   - Add logging and monitoring

4. **Final Evaluation**
   - Run full evaluation on test set
   - Compare against GPT-5 baseline
   - Optimize for blind test
   - Document results

**Deliverables**:
- ✅ 5-agent system working
- ✅ FastAPI deployed
- ✅ Full evaluation completed
- ✅ System optimized for competition

**Success Criteria**:
- Relevance: +15% vs GPT-5
- Citation accuracy: >90%
- Latency p95: <2s
- Cost per query: <$0.05

---

### Optional: Voice Support (Week 3, if ahead of schedule)

**Tasks**:
- Integrate OpenAI Whisper (STT)
- Integrate ElevenLabs or OpenAI TTS
- Test Hebrew voice recognition
- Add voice endpoints to API

**Bonus**: +5% of total score

---


## Success Metrics

### Target Performance (vs GPT-5 Baseline)

| Metric | Baseline (GPT-5) | Our Target | Improvement |
|--------|------------------|------------|-------------|
| **Relevance** (65%) | 75% | **90%** | +15% |
| **Citation Accuracy** (15%) | 85% | **95%** | +10% |
| **Efficiency** (10%) | 90% | **85%** | -5% (acceptable trade-off) |
| **Conversational Quality** (10%) | 80% | **90%** | +10% |
| **Total Weighted Score** | **78.5%** | **90.25%** | **+11.75%** |

### Key Performance Indicators (KPIs)

**Relevance Metrics**:
- Answer relevance (RAGAS): >0.9
- Context precision (RAGAS): >0.85
- User satisfaction (if available): >4.5/5

**Citation Metrics**:
- Citation accuracy: >90%
- Citation completeness: >95% (all claims cited)
- Source validity: 100% (all sources exist)

**Efficiency Metrics**:
- Latency p50: <1s
- Latency p95: <2s
- Latency p99: <3s
- Cost per query: <$0.05
- Throughput: >100 queries/minute

**Quality Metrics**:
- Hallucination rate: <5%
- Faithfulness (RAGAS): >0.9
- Clarity score: >0.8
- Professional tone: >0.9

---

## Risk Mitigation

### Technical Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Hebrew embedding quality poor** | High | Medium | Test early (Week 1), fallback to BGE-M3 |
| **Qwen 2.5 underperforms** | High | Low | Model-agnostic design, easy to swap to GPT-4o |
| **Scraping blocked/difficult** | High | Medium | Manual download backup, contact Harel |
| **Milvus performance issues** | Medium | Low | Use managed Milvus or Zilliz Cloud |
| **Latency too high** | Medium | Medium | Implement all optimization techniques |
| **5-agent complexity** | Medium | Medium | Start with 3-agent, expand if time permits |

### Project Risks

| Risk | Impact | Probability | Mitigation |
|------|--------|-------------|------------|
| **Scope creep** | High | High | Strict prioritization, skip voice/UI if needed |
| **Team member unavailable** | Medium | Medium | Clear documentation, modular design |
| **Infrastructure costs** | Low | Low | Use free tiers, Nebius credits |
| **Time pressure** | High | High | Weekly milestones, daily standups |

### Contingency Plans

**If behind schedule**:
1. **Week 1**: Skip GPT-5 baseline, focus on GPT-4o only
2. **Week 2**: Use simpler chunking (fixed 1024), skip hybrid search
3. **Week 3**: Implement 3-agent instead of 5-agent, skip voice/UI

**If ahead of schedule**:
1. **Week 2**: Add voice support
2. **Week 3**: Build web UI, add advanced features (query expansion, etc.)

---

## Milestones & Checkpoints

### Week 1 Milestones (Feb 1-7)

**Day 1-2** (Feb 1-2):
- ✅ Environment setup complete
- ✅ Milvus deployed
- ✅ Data scraping started

**Day 3-4** (Feb 3-4):
- ✅ 100+ documents scraped
- ✅ Evaluation framework working
- ✅ Reference questions loaded

**Day 5-7** (Feb 5-7):
- ✅ GPT-4o baseline complete
- ✅ Baseline report written
- ✅ Week 1 review meeting

**Checkpoint**: Can we beat the baseline? What are the failure patterns?

---

### Week 2 Milestones (Feb 8-14)

**Day 8-10** (Feb 8-10):
- ✅ Document processing pipeline working
- ✅ Milvus index created
- ✅ Embeddings generated

**Day 11-13** (Feb 11-13):
- ✅ Retrieval pipeline working
- ✅ Qwen 2.5 deployed on Nebius
- ✅ Answer generation working

**Day 14** (Feb 14):
- ✅ RAG system beats baseline
- ✅ <5% hallucination rate
- ✅ Week 2 review meeting

**Checkpoint**: Are we on track to beat GPT-5 by +15%?

---

### Week 3 Milestones (Feb 15-21)

**Day 15-17** (Feb 15-17):
- ✅ 5-agent system implemented
- ✅ Performance optimizations added
- ✅ FastAPI deployed

**Day 18-20** (Feb 18-20):
- ✅ Full evaluation complete
- ✅ System optimized
- ✅ Documentation finalized

**Day 21** (Feb 21):
- ✅ Final testing
- ✅ Presentation prepared
- ✅ Demo ready

**Checkpoint**: Ready for final presentation on Feb 22?

---

### Final Presentation (Feb 22)

**Agenda**:
1. **Problem Statement** (2 min)
2. **Architecture Overview** (5 min)
   - 5-agent system
   - Multi-strategy retrieval
   - Verification pipeline
3. **Live Demo** (5 min)
   - Show complex question handling
   - Highlight citations
   - Show cross-domain synthesis
4. **Results** (5 min)
   - Metrics comparison vs GPT-5
   - Latency and cost analysis
   - Failure analysis
5. **Lessons Learned** (3 min)
6. **Q&A** (5 min)

---

## Next Steps

### Immediate Actions (This Week)

1. **Review and Approve Design**
   - Team review of DESIGN.md
   - Discuss 5-agent vs 3-agent approach
   - Finalize tech stack decisions
   - Assign team roles

2. **Environment Setup**
   - Install dependencies (requirements.txt)
   - Deploy Milvus (docker-compose up)
   - Configure API keys (.env)
   - Test connections

3. **Data Collection Planning**
   - Identify Harel website structure
   - Plan scraping strategy
   - Set up data storage
   - Define metadata schema

4. **Evaluation Framework**
   - Install RAGAS
   - Load Reference_Questions.json
   - Define custom metrics
   - Create evaluation scripts

### Questions to Discuss

1. **Agent Architecture**: 5-agent (competitive) vs 3-agent (practical)?
2. **Team Roles**: Who owns which components?
3. **Daily Standups**: When and how often?
4. **Communication**: Slack, Discord, or other?
5. **Code Repository**: GitHub, GitLab, or other?
6. **Voice/UI Priority**: Worth pursuing or focus on core?
7. **Budget**: Any constraints on API costs?

### Decision Log

| Decision | Rationale | Date |
|----------|-----------|------|
| 5-agent architecture | Maximize competitiveness, beat rivals | Feb 15, 2026 |
| Qwen 2.5 72B (Nebius) | Best multilingual, cost-effective | Feb 15, 2026 |
| OpenAI embeddings | Proven Hebrew support | Feb 15, 2026 |
| 1024 base chunk size | Better context preservation | Feb 15, 2026 |
| Document-type aware chunking | Optimize for different content types | Feb 15, 2026 |
| All optimization techniques | Maximize performance | Feb 15, 2026 |
| RAGAS for evaluation | Industry standard for RAG | Feb 15, 2026 |
| Voice support: lower priority | Focus on core first | Feb 15, 2026 |

---

## Summary

**What We're Building**:
- Production-grade customer support chatbot for Harel Insurance
- 5-agent architecture with specialized roles
- Multi-strategy retrieval (semantic + keyword + hybrid)
- Grounded generation with inline citations
- Multi-stage verification to catch errors

**Why We'll Win**:
- ✅ Most teams will use simple RAG → we use sophisticated 5-agent pipeline
- ✅ Query decomposition handles complex questions better
- ✅ Multi-strategy retrieval improves recall and precision
- ✅ Cross-encoder re-ranking significantly boosts relevance
- ✅ Verification agent catches errors before submission
- ✅ All optimization techniques for <2s latency

**Expected Results**:
- Relevance: +15% vs GPT-5 baseline
- Citation accuracy: >90%
- Hallucination rate: <5%
- Latency p95: <2s
- **Total score: ~90% (vs 78.5% baseline)**

**Timeline**: 3 weeks (Feb 1-21, 2026)
**Presentation**: Feb 22, 2026

---

**Ready to build? Let's win this! 🏆**

