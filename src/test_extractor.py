import os
import shutil
from extraction import PDFExtractor
from splitter import MarkdownSplitter
from index_manger import MultiVectorIndexManager
from rag_chain import RAGChain

def main():
    pdf = "data/data.pdf"
    chroma_dir = "data/chroma"
    docstore_dir = "data/docstore"
    
    if not os.path.exists(pdf):
        print(f"Error: Could not find PDF at {pdf}")
        return

    skip_extraction = False
   
    index_exists = (os.path.exists(chroma_dir) and len(os.listdir(chroma_dir)) > 0)
    
    if index_exists:
        choice = input("\nExisting index found! Do you want to load it? (y/n, default y): ").strip().lower()
        if choice == 'n':
            print("\nForcing a rebuild of the index...")
            # Safely clear out old physical database directories to prevent stale mismatches
            try:
                if os.path.exists(chroma_dir):
                    shutil.rmtree(chroma_dir)
                if os.path.exists(docstore_dir):
                    shutil.rmtree(docstore_dir)
                print("Successfully cleared existing Chroma and Docstore directories.")
            except Exception as e:
                print(f"Warning: Could not clear database directories: {e}")
        else:
            print("\nLoading embeddings from local storage.")
            print("Skipping PDF extraction and embedding to save time and tokens.")
            skip_extraction = True
    else:
        print("\nNo existing index found.")

    print("Initializing components...")
    # Now safe to initialize the index manager (will create empty DB directories if deleted above)
    index_manager = MultiVectorIndexManager()

    if not skip_extraction:
        print("\nStarting extraction process...")
        extractor = PDFExtractor(pdf_path=pdf)
        splitter = MarkdownSplitter()

        print("\nExtracting PDF text to Markdown...")
        markdown = extractor.extract_markdown()
        chunks = splitter.split(markdown)
        print(f"Created {len(chunks)} text chunks.")

        print("\nExtracting images...")
        images = extractor.extract_images()
        print(f"Extracted {len(images)} meaningful images.")

        print("\nRebuilding vector index...")
        index_manager.reset_index()
        
        print("Indexing text...")
        index_manager.index_text_chunks(chunks)

        print("Indexing images (this might take a moment to generate summaries)...")
        index_manager.index_images(images)

    print("\nSystem ready! Starting RAG Chain.")
    rag = RAGChain(index_manager=index_manager)

    while True:
        question = input("\nEnter your question (or type 'exit' to quit): ")
        if question.lower() == 'exit':
            break

        response = rag.ask(question)
        print(f"\nResponse:\n{response}")

if __name__ == "__main__":
    main()