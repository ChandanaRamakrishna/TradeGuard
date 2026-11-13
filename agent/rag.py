import re
from functools import lru_cache

from langchain_text_splitters import RecursiveCharacterTextSplitter
from risk_engine.io import ROOT

POLICY_PATH = ROOT / "data" / "policies" / "risk_policy.txt"
DB_PATH = ROOT / ".chroma"
POLICY_SOURCE = "data/policies/risk_policy.txt"
COLLECTION_NAME = "tradeguard_policy_sections_v1"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
_SECTION_HEADER = re.compile(r"(?m)^(\d+)\.\s+(.+?)\s*$")


def _policy_sections():
    if not POLICY_PATH.is_file():
        return []
    text = POLICY_PATH.read_text(encoding="utf-8")
    matches = list(_SECTION_HEADER.finditer(text))
    sections = []
    if matches and text[:matches[0].start()].strip():
        sections.append(("Preamble", text[:matches[0].start()].strip()))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        title = f"{match.group(1)}. {match.group(2)}"
        body = text[match.end():end].strip()
        sections.append((title, f"{title}\n{body}".strip()))
    if not sections and text.strip():
        sections.append(("Policy", text.strip()))
    return sections


def _policy_chunks():
    from langchain_core.documents import Document

    splitter = RecursiveCharacterTextSplitter(chunk_size=550, chunk_overlap=80)
    chunks = []
    for section, text in _policy_sections():
        section_doc = Document(
            page_content=text,
            metadata={"source": POLICY_SOURCE, "section": section,
                      "type": "synthetic_sample_policy"},
        )
        chunks.extend(splitter.split_documents([section_doc]))
    return chunks


@lru_cache(maxsize=2)
def _embeddings(local_files_only: bool):
    from langchain_huggingface import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"local_files_only": local_files_only},
    )


def build_policy_store():
    """Build or replace the versioned Chroma collection from the local policy."""
    import chromadb
    from langchain_chroma import Chroma

    chunks = _policy_chunks()
    if not chunks:
        raise FileNotFoundError(f"Policy document is missing or empty: {POLICY_PATH}")
    DB_PATH.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(DB_PATH))
    names = {getattr(collection, "name", collection) for collection in client.list_collections()}
    if COLLECTION_NAME in names:
        client.delete_collection(name=COLLECTION_NAME)
    embeddings = _embeddings(local_files_only=False)
    return Chroma.from_documents(
        chunks,
        embeddings,
        collection_name=COLLECTION_NAME,
        persist_directory=str(DB_PATH),
    )


def _section_from_doc(doc):
    section = doc.metadata.get("section")
    if section:
        return section
    match = _SECTION_HEADER.search(doc.page_content)
    return f"{match.group(1)}. {match.group(2)}" if match else "Policy"


def _normalize_text(text):
    return " ".join(text.casefold().split())


def _distinct_results(docs, k: int, retrieval_method: str):
    results = []
    seen_sections = set()
    seen_text = set()
    for doc in docs:
        text_key = _normalize_text(doc.page_content)
        section = _section_from_doc(doc)
        if text_key in seen_text or section in seen_sections:
            continue
        seen_text.add(text_key)
        seen_sections.add(section)
        results.append({
            "text": doc.page_content,
            "source": doc.metadata.get("source", POLICY_SOURCE),
            "section": section,
            "retrieval_method": retrieval_method,
        })
        if len(results) >= k:
            break
    return results


def _lexical_retrieve(question: str, k: int):
    from langchain_core.documents import Document

    stop_words = {
        "a", "an", "and", "are", "for", "from", "how", "is", "of", "or",
        "the", "to", "what", "when", "where", "which", "with", "does", "say",
    }
    terms = set(re.findall(r"[a-z0-9]+", question.casefold())) - stop_words
    ranked = []
    for index, (section, text) in enumerate(_policy_sections()):
        section_tokens = set(re.findall(r"[a-z0-9]+", section.casefold()))
        body_tokens = set(re.findall(r"[a-z0-9]+", text.casefold()))
        score = 3 * len(terms & section_tokens) + len(terms & body_tokens)
        if score:
            ranked.append((score, index, Document(
                page_content=text,
                metadata={"source": POLICY_SOURCE, "section": section},
            )))
    ranked.sort(key=lambda item: (-item[0], item[1]))
    return _distinct_results([doc for _, _, doc in ranked], k, "local_keyword_fallback")


def retrieve_policy(question: str, k: int = 3):
    """Return relevant, distinct policy sections with source attribution.

    Chroma is preferred when its versioned collection is available. Missing,
    stale, or failed vector retrieval falls back to local section keyword search.
    """
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question must be a non-empty string.")
    if isinstance(k, bool) or not isinstance(k, int) or k <= 0:
        raise ValueError("k must be a positive integer.")

    if DB_PATH.exists():
        try:
            from langchain_chroma import Chroma
            embeddings = _embeddings(local_files_only=True)
            store = Chroma(
                collection_name=COLLECTION_NAME,
                persist_directory=str(DB_PATH),
                embedding_function=embeddings,
            )
            count = store._collection.count()
            if count:
                candidates = store.similarity_search(question, k=min(count, max(k * 4, 12)))
                results = _distinct_results(candidates, k, "chroma")
                if results:
                    return results
        except Exception:
            # Chroma and model-loading failures must not prevent local policy use.
            pass
    return _lexical_retrieve(question, k)
