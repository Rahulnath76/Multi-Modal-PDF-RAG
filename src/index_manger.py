from langchain_chroma import Chroma
from langchain_classic.storage import LocalFileStore
from langchain_classic.storage._lc_store import create_kv_docstore
from langchain_classic.retrievers.multi_vector import MultiVectorRetriever
from langchain_huggingface import HuggingFaceEndpointEmbeddings
from langchain_core.documents import Document
from image_summarizer import ImageSummarizer
import uuid
from dotenv import load_dotenv
import os

load_dotenv()


class MultiVectorIndexManager:
    def __init__(self, persist_directory="data/chroma", docstore_directory="data/docstore", collection_name="multi_vector_index"):
        self.id_key = "doc_id"
        self.persist_directory = persist_directory
        self.docstore_directory = docstore_directory
        self.collection_name = collection_name
        self.embeddings = HuggingFaceEndpointEmbeddings(
            model="BAAI/bge-base-en-v1.5")

        os.makedirs(persist_directory, exist_ok=True)
        os.makedirs(docstore_directory, exist_ok=True)

        self.vector_store = Chroma(
            collection_name=collection_name,
            persist_directory=persist_directory,
            embedding_function=self.embeddings,
        )

        fs = LocalFileStore(docstore_directory)
        self.docstore = create_kv_docstore(fs)

        self.retriever = MultiVectorRetriever(
            vectorstore=self.vector_store,
            docstore=self.docstore,
            id_key=self.id_key,
            search_kwargs={"k": 10}
        )

        self.image_summarizer = ImageSummarizer()

    def reset_index(self):
        try:
            self.vector_store.delete_collection()
        except:
            pass

        self.vector_store = Chroma(
            collection_name=self.collection_name,
            persist_directory=self.persist_directory,
            embedding_function=self.embeddings,
        )

        for key in self.docstore.yield_keys():
            self.docstore.mdelete([key])

        self.retriever = MultiVectorRetriever(
            vectorstore=self.vector_store,
            docstore=self.docstore,
            id_key=self.id_key,
            search_kwargs={"k": 10}
        )

    def index_text_chunks(self, text_chunks):
        vector_docs = []
        docstore_pairs = []

        for chunk_number, chunk in enumerate(text_chunks, start=1):
            doc_id = str(uuid.uuid4())

            metadata = chunk.metadata.copy()
            metadata[self.id_key] = doc_id
            metadata['type'] = 'text'
            metadata['chunk_number'] = chunk_number

            vector_docs.append(
                Document(page_content=chunk.page_content, metadata=metadata))

            docstore_pairs.append(
                (doc_id, Document(page_content=chunk.page_content, metadata=metadata)))

        if not vector_docs:
            print("Indexed 0 text chunks")
            return

        self.vector_store.add_documents(vector_docs)
        self.docstore.mset(docstore_pairs)
        print(f"Indexed {len(vector_docs)} text chunks")

    def index_images(self, images):
        vector_docs = []
        docstore_pairs = []

        for image in images:
            doc_id = str(uuid.uuid4())

            summary = self.image_summarizer.summarize_image(image.get("bytes"))
            summary = str(summary).strip()

            metadata = {
                self.id_key: doc_id,
                "type": "image",
                "page": image.get("page"),
            }

            vector_docs.append(
                Document(page_content=summary, metadata=metadata))
            
            doc_content = f"[IMAGE SUMMARY - Page {image.get('page')}]:\n{summary}"
            docstore_pairs.append(
                (doc_id, Document(page_content=doc_content, metadata=metadata)))

        if not vector_docs:
            print("Indexed 0 image chunks")
            return

        self.vector_store.add_documents(vector_docs)
        self.docstore.mset(docstore_pairs)
        print(f"Indexed {len(vector_docs)} image chunks")
