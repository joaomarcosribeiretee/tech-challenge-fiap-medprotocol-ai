"""Modelo de embeddings local (intfloat/multilingual-e5-small).

O E5 foi treinado com prefixos: "query: " para perguntas e "passage: " para trechos.
Eles são aplicados exatamente uma vez, via `prompt` do sentence-transformers:
- encode_kwargs       -> embed_documents (indexação)
- query_encode_kwargs -> embed_query (consultas)
O modelo não define prompts padrão (verificado na revisão fixada), então não há prefixo
automático adicional. O HuggingFaceEmbeddings troca "\\n" por " " antes de codificar;
`make_token_counter` mede o texto exatamente nessa forma.
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings

from medprotocol.rag.chunking import TokenCounter

MODEL_NAME = "intfloat/multilingual-e5-small"
MODEL_REVISION = "614241f622f53c4eeff9890bdc4f31cfecc418b3"
MAX_TOKENS = 512
QUERY_PREFIX = "query: "
PASSAGE_PREFIX = "passage: "


def make_embeddings(device: str = "cpu") -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(
        model_name=MODEL_NAME,
        model_kwargs={"device": device, "revision": MODEL_REVISION},
        encode_kwargs={"prompt": PASSAGE_PREFIX, "normalize_embeddings": True},
        query_encode_kwargs={"prompt": QUERY_PREFIX, "normalize_embeddings": True},
    )


def make_token_counter(embeddings: HuggingFaceEmbeddings) -> TokenCounter:
    """Conta tokens do trecho como o modelo o recebe (prefixo + tokens especiais)."""
    tokenizer = embeddings._client.tokenizer  # mesmo tokenizer usado pelo encode

    def count(text: str) -> int:
        prepared = PASSAGE_PREFIX + text.replace("\n", " ")
        return len(tokenizer(prepared, add_special_tokens=True, truncation=False)["input_ids"])

    return count
