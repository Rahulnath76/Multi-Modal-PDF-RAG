from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

class MarkdownSplitter:
    def __init__(self, chunk_size=1000, chunk_overlap=200):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

        self.header_splitter = MarkdownHeaderTextSplitter(
            headers_to_split_on=[
                ("#", "h1"),
                ("##", "h2"),
                ("###", "h3"),
            ],
        )

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
        )

    def split(self, markdown_text):
        header_docs = self.header_splitter.split_text(markdown_text)
        final_docs = self.text_splitter.split_documents(header_docs)
        
        return final_docs