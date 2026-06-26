from langchain_groq import ChatGroq
from langchain_core.messages import SystemMessage, HumanMessage

RAG_PROMPT = """You are an expert research assistant answering questions based on a provided document.
Use ONLY the supplied context below to answer the question. 

Context:
{context}

Question: {question}
Answer:"""

class RAGChain:
    def __init__(self, index_manager):
        self.index_manager = index_manager
        self.model = ChatGroq(model="openai/gpt-oss-120b")

    def ask(self, question):
        retrieved_docs = self.index_manager.retriever.invoke(question)
        
        print(f"\n[DEBUG] Retrieved {len(retrieved_docs)} chunks from the database.")
        
        # Safely format the retrieved LangChain Document objects
        formatted_context = []
        for idx, doc in enumerate(retrieved_docs):
            doc_type = doc.metadata.get("type", "unknown text")
            formatted_context.append(f"--- Context {idx + 1} ({doc_type}) ---\n{doc.page_content}")
            
            # Print a snippet to the terminal so you can see what the LLM sees
            print(f"[DEBUG] Chunk {idx + 1} preview: {doc.page_content[:100]}...")

        context_string = "\n\n".join(formatted_context)

        prompt = RAG_PROMPT.format(context=context_string, question=question)

        response = self.model.invoke([
            HumanMessage(content=prompt)
        ])

        return response.content