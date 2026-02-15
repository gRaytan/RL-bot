# Phase 1: Baseline Evaluation Report

**Date:** 2026-02-15  
**Evaluator:** Automated Baseline Runner  
**Dataset:** 20 Hebrew insurance questions across 5 domains

---

## Executive Summary

| Model | Strategy | Accuracy | Avg Latency | Notes |
|-------|----------|----------|-------------|-------|
| **Qwen3-235B** | basic | 50% | 5.8s | Verbose, confident but wrong |
| **Qwen3-235B** | strict | **55%** | 0.7s | Says "לא בטוח" when unsure ✅ |
| **Llama-3.3-70B** | basic | 40% | 22.4s | Slowest, most hallucinations |
| **DeepSeek-V3** | strict | 45% | 1.5s | Good on business domain |

**Key Finding:** Without RAG, even the best models achieve only **50-55% accuracy** on domain-specific insurance questions.

---

## Per-Domain Analysis

| Domain | Qwen3 Basic | Qwen3 Strict | Llama-3.3 | DeepSeek-V3 |
|--------|-------------|--------------|-----------|-------------|
| **דירה (Apartment)** | 71% (5/7) | **100% (7/7)** | 57% (4/7) | 57% (4/7) |
| **רכב (Car)** | 67% (2/3) | 67% (2/3) | 67% (2/3) | 67% (2/3) |
| **נסיעות (Travel)** | 50% (1/2) | 50% (1/2) | 50% (1/2) | 50% (1/2) |
| **בריאות (Health)** | 33% (1/3) | 0% (0/3) | 0% (0/3) | 0% (0/3) |
| **עסקים (Business)** | 20% (1/5) | 20% (1/5) | 20% (1/5) | 40% (2/5) |

### Observations

1. **Apartment domain** performs best - likely more general knowledge available
2. **Health domain** is worst - requires specific policy details
3. **Business domain** has specific numbers/dates that models hallucinate

---

## Failure Analysis

### Type 1: Hallucinated Facts (High Risk)

Models confidently provide **wrong specific information**:

| Question | Expected | Model Said |
|----------|----------|------------|
| מספר הטלפון של מוקד תביעות | 03-9294000 | 1-700-50-60-60 ❌ |
| תקופת התיישנות | 3 שנים | 7 שנים ❌ |
| דוד שמש 300 ליטר | לא מכוסה | מכוסה ❌ |

**Impact:** Customer gets wrong information, potential legal issues.

### Type 2: Missing Domain Knowledge

Models don't know Harel-specific policies:

- Coverage exclusions for specific items
- Policy-specific limits and conditions
- Internal phone numbers and procedures

### Type 3: Citation Failures (100%)

**No model provided any citations** to source documents.

---

## Strict Strategy Impact

The "strict" prompt strategy improved behavior:

| Metric | Basic | Strict | Change |
|--------|-------|--------|--------|
| Accuracy | 50% | 55% | +5% |
| "לא בטוח" responses | 0 | 4 | ✅ Better |
| Hallucinated numbers | 5 | 2 | ✅ Reduced |
| Latency | 5.8s | 0.7s | ✅ 8x faster |

**Conclusion:** Strict prompting helps, but RAG is essential for domain accuracy.

---

## Recommendations for Phase 2

### 1. RAG is Essential
- 50% baseline accuracy is unacceptable for production
- Domain-specific knowledge must come from indexed documents

### 2. Focus Areas for RAG
- **Business domain** (20% accuracy) - needs most improvement
- **Health domain** (0-33% accuracy) - critical for customer trust
- **Specific numbers** (phone, dates, limits) - must come from documents

### 3. Model Selection
- **Qwen3-235B** with strict prompting is best baseline
- Consider **DeepSeek-V3** for cost efficiency (similar accuracy, faster)

### 4. Evaluation Metrics to Track
- [ ] Answer correctness (currently 50-55%)
- [ ] Hallucination rate (currently ~25%)
- [ ] Citation rate (currently 0%)
- [ ] "I don't know" rate (should increase with RAG)

---

## Next Steps

1. **Implement RAG pipeline** with Harel insurance documents
2. **Re-run evaluation** with retrieved context
3. **Target:** 85%+ accuracy with proper citations
4. **Add RAGAS metrics** for deeper analysis

---

*Report generated automatically by baseline evaluation framework*

