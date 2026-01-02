# Phase 2 Session Prompt

Copy and paste this to start a new Claude Code session:

---

## Context

I'm building an Encyclopedia Evolution Knowledge Graph to track how encyclopedic knowledge evolved from 1704-1911.

**Read these files first:**
- `PROJECT.md` - Overall project documentation
- `PARSER_STATUS.md` - Current parser development status

**Phase 1 is complete.** I created `encyclopedia_parser/` module that:
- Extracts articles from OLMoCR markdown/JSONL output
- Classifies articles as dictionary/treatise/biographical/geographical/cross_reference
- Tested on 2nd edition: 2,471 articles extracted with good accuracy

## Task: Phase 2 - Treatise Chunking for RAG

Implement semantic chunking for long treatise articles so they can be embedded for vector search.

### Requirements

1. **Create `encyclopedia_parser/chunkers.py`** with:
   - `chunk_treatise(article: Article) -> list[TextChunk]` function
   - Use LangChain's SemanticChunker for treatises (>5000 chars)
   - Short articles return single chunk
   - Track chunk metadata (parent headword, index, char positions)

2. **Integration:**
   - Add chunking to the extraction pipeline
   - Option to chunk on-demand vs during extraction

3. **Cost tracking:**
   - Log embedding API calls (OpenAI embeddings used by SemanticChunker)
   - Estimate cost per volume

### Dependencies to install first:
```bash
pip install langchain langchain-experimental langchain-openai openai
```

### Test data:
```python
# Get treatises from 2nd edition vol 1
from encyclopedia_parser import parse_markdown_file, classify_articles
articles = parse_markdown_file("ocr_results/britannica_pipeline_batch/britannica_nls_144850370.md", 1778)
treatises = [a for a in classify_articles(articles) if a.article_type == "treatise"]
# Should have ~81 treatises to test with
```

### Reference: TextChunk model already exists in models.py:
```python
class TextChunk(BaseModel):
    text: str
    index: int
    parent_headword: str
    edition_year: int
    char_start: int
    char_end: int
    section_title: Optional[str] = None
```

Please implement Phase 2 following the plan in `/home/jic823/.claude/plans/ancient-forging-diffie.md`

---
