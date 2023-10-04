import copy
from langchain.pydantic_v1 import Field
from langchain.load.serializable import Serializable

class Document(Serializable):
    """Class for storing a piece of text and associated metadata."""

    page_content: str
    """String text."""
    metadata: dict = Field(default_factory=dict)
    """Arbitrary metadata about the page content (e.g., source, relationships to other
        documents, etc.).
    """

    @classmethod
    def is_lc_serializable(cls) -> bool:
        """Return whether this class is serializable."""
        return True


def create_documents(texts, metadatas):
    _metadatas = metadatas or [{}] * len(texts)
    documents = []
    for chunk in split_text(texts):
        metadata = copy.deepcopy(_metadatas[0])
        new_doc = Document(page_content=chunk, metadata=metadata)
        documents.append(new_doc)
    return documents   

def split_text(documents):
    f=[]
    for doc in str(documents).split('Статья'):
        f.append('Статья' + doc)
    return f

def split_documents(documents):
    """Split documents."""
    texts, metadatas = [], []
    for doc in documents:
        texts.append(doc.page_content)
        metadatas.append(doc.metadata)
    return create_documents(texts, metadatas=metadatas) 