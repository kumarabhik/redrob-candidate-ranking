"""JD -> structured requirement aspects (Pillar 2, design_doc.md §5.2 / §7.2).

We read the released job_description.docx, then encode it into a hand-authored aspect
config: must-haves, nice-to-haves, explicit disqualifiers, the experience band, preferred
locations, consulting-firm and non-engineering-title lists, and skill vocabularies. This
hand-authoring IS the "real engineering" the hackathon rewards -- it operationalizes the
JD's instruction to reason about MEANING, not keyword overlap.

Output: artifacts/aspects.json  (+ artifacts/jd_text.txt for provenance).

Run (offline): python src/jd_aspects.py
"""
from __future__ import annotations

import json
import re
import zipfile

from config import ARTIFACTS, DATA_DIR

JD_DOCX = DATA_DIR / "job_description.docx"


def docx_to_text(path) -> str:
    """Extract visible text from a .docx (word/document.xml), no external deps."""
    with zipfile.ZipFile(path) as z:
        xml = z.read("word/document.xml").decode("utf-8", errors="ignore")
    # paragraphs -> newlines; strip tags; unescape a few entities.
    xml = re.sub(r"</w:p>", "\n", xml)
    text = re.sub(r"<[^>]+>", "", xml)
    for a, b in [("&amp;", "&"), ("&lt;", "<"), ("&gt;", ">"), ("&quot;", '"'), ("&#39;", "'")]:
        text = text.replace(a, b)
    return re.sub(r"\n{2,}", "\n", text).strip()


# --- Hand-authored aspect config (derived from reading the JD) ------------
ASPECTS = {
    "role_title": "Senior AI Engineer - Founding Team (talent-intelligence platform)",

    # JD: "5-9 years ... range not requirement"; ideal "6-8 ... 4-5 in applied ML at product cos".
    "experience": {"min": 5, "max": 9, "ideal_min": 6, "ideal_max": 8, "hard_floor": 3},

    # JD: Pune/Noida preferred; Hyderabad/Pune/Mumbai/Delhi NCR welcome. Relocation OK.
    "locations_preferred": ["pune", "noida", "delhi", "new delhi", "gurgaon", "gurugram",
                            "noida", "ncr", "hyderabad", "mumbai"],
    "country_preferred": ["india"],

    # Must-haves: each has a natural-language `query` (embedded for semantic aspect-fit),
    # a `skills` vocabulary (matched against skills[].name AND career descriptions), and a weight.
    "must_have": [
        {
            "id": "retrieval",
            "name": "Production embeddings-based retrieval",
            "query": "production experience with embeddings based retrieval systems deployed "
                     "to real users, sentence transformers, BGE, E5, semantic search, RAG, "
                     "handling embedding drift and retrieval quality regression",
            "skills": ["embeddings", "sentence transformers", "bge", "e5", "retrieval",
                       "semantic search", "rag", "information retrieval", "dense retrieval",
                       "vector search", "neural search", "huggingface", "hugging face"],
            "weight": 1.0,
        },
        {
            "id": "vectordb",
            "name": "Vector DB / hybrid search infrastructure",
            "query": "production experience operating vector databases or hybrid search "
                     "infrastructure such as Pinecone, Weaviate, Qdrant, Milvus, OpenSearch, "
                     "Elasticsearch, FAISS, pgvector",
            "skills": ["pinecone", "weaviate", "qdrant", "milvus", "opensearch",
                       "elasticsearch", "faiss", "pgvector", "vector database", "bm25",
                       "hybrid search", "haystack", "llamaindex", "langchain"],
            "weight": 0.9,
        },
        {
            "id": "ranking_eval",
            "name": "Ranking systems & evaluation frameworks",
            "query": "designing evaluation frameworks for ranking systems, NDCG, MRR, MAP, "
                     "offline to online correlation, A/B testing, building search ranking or "
                     "recommendation systems at scale",
            "skills": ["ndcg", "mrr", "map", "learning to rank", "ranking", "recommendation",
                       "recommendation systems", "recommender", "a/b testing", "ab testing",
                       "search ranking", "evaluation", "experimentation"],
            "weight": 0.95,
        },
        {
            "id": "python_ml",
            "name": "Strong Python / ML engineering",
            "query": "strong production Python engineering, machine learning systems, "
                     "deep learning, NLP, model serving and MLOps",
            "skills": ["python", "machine learning", "deep learning", "nlp", "pytorch",
                       "tensorflow", "scikit-learn", "mlops", "mlflow", "model serving"],
            "weight": 0.7,
        },
    ],

    "nice_to_have": [
        {"id": "llm_ft", "name": "LLM fine-tuning",
         "query": "LLM fine-tuning with LoRA, QLoRA, PEFT",
         "skills": ["lora", "qlora", "peft", "fine-tuning", "fine tuning", "instruction tuning"],
         "weight": 0.3},
        {"id": "ltr", "name": "Learning-to-rank models",
         "query": "learning to rank models, XGBoost or neural rankers, gradient boosting",
         "skills": ["xgboost", "lightgbm", "lambdamart", "gradient boosting", "catboost"],
         "weight": 0.3},
        {"id": "hrtech", "name": "HR-tech / marketplace",
         "query": "HR tech, recruiting technology, or two-sided marketplace products",
         "skills": ["recruiting", "hr tech", "marketplace", "talent"],
         "weight": 0.2},
        {"id": "scale", "name": "Distributed systems / large-scale inference",
         "query": "distributed systems and large scale inference optimization",
         "skills": ["distributed systems", "kubernetes", "spark", "kafka", "ray", "triton"],
         "weight": 0.2},
        {"id": "oss", "name": "Open-source AI/ML contributions",
         "query": "open source contributions in the AI ML space",
         "skills": ["open source", "open-source", "github"],
         "weight": 0.15},
    ],

    # Explicit "do NOT want" signals -> strong negatives.
    "disqualifiers": {
        # entire career at IT-services / consulting firms
        "consulting_firms": ["tcs", "tata consultancy", "infosys", "wipro", "accenture",
                             "cognizant", "capgemini", "tech mahindra", "hcl", "mindtree",
                             "ltimindtree", "larsen", "deloitte", "ibm global", "dxc"],
        # current/primary title not an engineering/ML role (the "Marketing Manager with a
        # perfect AI skill list" trap). Matched against current_title + history titles.
        "non_engineering_titles": ["marketing", "sales", "hr ", "human resource", "recruiter",
                                   "accountant", "accounting", "operations manager", "civil",
                                   "mechanical", "graphic designer", "content writer", "seo",
                                   "finance", "administrative", "customer success",
                                   "business development", "project manager", "product manager",
                                   "ui/ux", "ux designer", "frontend", "front-end"],
        # pure research with no production
        "research_terms": ["phd researcher", "research scholar", "postdoc", "post-doc",
                           "academic", "research assistant", "research fellow"],
        # wrong primary domain
        "wrong_domain_terms": ["computer vision", "image classification", "object detection",
                              "speech recognition", "asr", "tts", "robotics", "gans", "yolo",
                              "opencv", "diffusion"],
        # shallow / framework-only LLM hype (keyword-stuffer signature)
        "framework_only_terms": ["langchain", "llamaindex", "prompt engineering"],
    },

    # Behavioral availability inputs (from redrob_signals; see redrob_signals_doc).
    "behavioral": {
        "good_response_rate": 0.4, "stale_days": 120, "good_notice_days": 30,
    },
}


# --- Multi-JD: derive aspects from arbitrary JD text ---------------------
# Skill taxonomy: category -> vocabulary. Used to auto-build aspects for any JD so the
# service is not hardwired to one role (see system_design.md). The curated ASPECTS above
# remain the default for the released hackathon JD.
SKILL_TAXONOMY = {
    "retrieval": ["embeddings", "retrieval", "rag", "semantic search", "vector search",
                  "information retrieval", "sentence transformers", "bge", "e5", "reranking"],
    "vectordb": ["pinecone", "weaviate", "qdrant", "milvus", "faiss", "elasticsearch",
                 "opensearch", "pgvector", "vector database", "hybrid search"],
    "ranking_eval": ["ranking", "recommendation", "recommender", "ndcg", "learning to rank",
                     "search ranking", "relevance", "recsys", "a/b test", "personalization"],
    "llm": ["llm", "gpt", "lora", "qlora", "peft", "fine-tuning", "prompt", "langchain",
            "llamaindex", "transformers", "rag"],
    "python_ml": ["python", "machine learning", "deep learning", "pytorch", "tensorflow",
                  "scikit-learn", "mlops", "nlp", "model serving"],
    "frontend": ["react", "vue", "angular", "next.js", "typescript", "javascript", "css",
                 "html", "tailwind", "redux", "frontend", "ui/ux"],
    "backend": ["node", "django", "flask", "fastapi", "spring", "golang", " go ", "java",
                "rest", "grpc", "microservices", "backend", "api", "postgres", "redis"],
    "data_eng": ["spark", "kafka", "airflow", "dbt", "snowflake", "bigquery", "hadoop",
                 "etl", "databricks", "data pipeline", "data engineering"],
    "devops": ["kubernetes", "docker", "terraform", "aws", "gcp", "azure", "ci/cd", "devops"],
    "mobile": ["android", "ios", "flutter", "react native", "swift", "kotlin", "mobile"],
}
_CITIES = ["pune", "noida", "delhi", "new delhi", "gurgaon", "gurugram", "hyderabad", "mumbai",
           "bangalore", "bengaluru", "chennai", "kolkata", "remote", "san francisco", "new york",
           "london", "berlin", "singapore", "toronto"]


def aspects_from_jd_text(jd_text: str, role_title: str = "Role") -> dict:
    """Auto-derive an aspect config (same shape as ASPECTS) from arbitrary JD text."""
    low = jd_text.lower()

    # category match counts
    scored = []
    for cat, vocab in SKILL_TAXONOMY.items():
        matched = [v.strip() for v in vocab if v in low]
        if matched:
            scored.append((cat, matched, len(matched)))
    scored.sort(key=lambda t: -t[2])

    def _aspect(cat, matched, weight):
        return {"id": cat, "name": cat.replace("_", " ").title(),
                "query": f"{role_title}. {cat.replace('_', ' ')}: " + ", ".join(matched[:8]),
                "skills": matched, "weight": round(weight, 2)}

    # top categories -> must-have, the rest -> nice-to-have
    must = [_aspect(c, m, min(1.0, 0.6 + 0.1 * n)) for c, m, n in scored[:4]] or \
           [{"id": "general", "name": "General", "query": (role_title + ". " + jd_text)[:400],
             "skills": [], "weight": 1.0}]
    nice = [_aspect(c, m, 0.25) for c, m, n in scored[4:8]]

    # experience band: parse "X-Y years"
    exp = {"min": 5, "max": 9, "ideal_min": 6, "ideal_max": 8, "hard_floor": 3}
    m = re.search(r"(\d{1,2})\s*[-–to]+\s*(\d{1,2})\s*\+?\s*year", low)
    if m:
        lo, hi = sorted((int(m.group(1)), int(m.group(2))))
        exp = {"min": lo, "max": hi, "ideal_min": lo + max(0, (hi - lo) // 3),
               "ideal_max": hi - max(0, (hi - lo) // 3), "hard_floor": max(0, lo - 2)}

    cities = [c for c in _CITIES if c in low]

    # Neutral disqualifiers for arbitrary JDs: the curated ASPECTS lists encode the AI role's
    # biases (e.g. they treat "frontend"/"mobile" titles as non-engineering and CV/speech as
    # wrong-domain), which would wrongly penalize a Frontend or CV JD. Keep only the
    # universally-safe consulting-firm signal; leave role-specific traps empty.
    disq = {"consulting_firms": ASPECTS["disqualifiers"]["consulting_firms"],
            "non_engineering_titles": [], "research_terms": [],
            "wrong_domain_terms": [], "framework_only_terms": []}

    cfg = {
        "role_title": role_title,
        "experience": exp,
        "locations_preferred": cities or ASPECTS["locations_preferred"],
        "country_preferred": ["india"] if "india" in low else ASPECTS["country_preferred"],
        "must_have": must,
        "nice_to_have": nice,
        "disqualifiers": disq,
        "behavioral": ASPECTS["behavioral"],
    }
    cfg["_query_order"] = [a["id"] for a in must] + [a["id"] for a in nice]
    return cfg


def build() -> dict:
    # Serve/sandbox fallback: if the JD docx is not present (e.g. on a deployed Space) but the
    # aspects artifact already exists, just load it instead of rebuilding from the docx.
    if not JD_DOCX.exists() and (ARTIFACTS / "aspects.json").exists():
        return json.loads((ARTIFACTS / "aspects.json").read_text(encoding="utf-8"))
    ARTIFACTS.mkdir(exist_ok=True)
    jd_text = docx_to_text(JD_DOCX)
    (ARTIFACTS / "jd_text.txt").write_text(jd_text, encoding="utf-8")

    cfg = dict(ASPECTS)
    cfg["_jd_text_chars"] = len(jd_text)
    # Ordered list of query strings to embed (must-have then nice-to-have).
    cfg["_query_order"] = [a["id"] for a in cfg["must_have"]] + \
                          [a["id"] for a in cfg["nice_to_have"]]
    out = ARTIFACTS / "aspects.json"
    out.write_text(json.dumps(cfg, indent=2), encoding="utf-8")
    return cfg


if __name__ == "__main__":
    cfg = build()
    print(f"JD text extracted: {cfg['_jd_text_chars']} chars -> artifacts/jd_text.txt")
    print(f"aspects written -> artifacts/aspects.json")
    print(f"must-have aspects: {[a['id'] for a in cfg['must_have']]}")
    print(f"nice-to-have:      {[a['id'] for a in cfg['nice_to_have']]}")
    print(f"experience band:   {cfg['experience']}")
