
from typing import List
import time

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains.conversational_retrieval.base import ConversationalRetrievalChain
from langchain.memory.buffer import ConversationBufferMemory
from langchain_core.documents import Document
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage
from src.config import Config


class ChatManager:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.memory = None
        self.chain = None
        self.llm = None
        self._initialize_components()

    def _initialize_components(self):
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )

        if not self.api_key:
            raise ValueError("Gemini API key is missing.")

        self.llm = ChatGoogleGenerativeAI(
            model=Config.MODEL_NAME,
            google_api_key=self.api_key,
            temperature=0.7,
            max_output_tokens=2048
        )

        print(f"Gemini model configured: {Config.MODEL_NAME}")

    def _create_chain(self, retriever):
        system_template = """You are a helpful assistant that answers
        questions based on the provided context.

        If you cannot find the answer in the context, acknowledge that.
        Always cite your sources when information comes from the context.

        Context:
        {context}
        """

        qa_prompt = PromptTemplate(
            input_variables=["context", "question"],
            template=system_template + "\nQuestion: {question}"
        )

        self.chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            return_source_documents=True,
            combine_docs_chain_kwargs={"prompt": qa_prompt},
            verbose=False
        )

        return self.chain

    def _fallback_generation(self, query: str, context_text: str):
        messages = [
            SystemMessage(
                content=f"""You are a helpful assistant.
                Answer using the following PDF context:

                {context_text}
                """
            ),
            HumanMessage(content=query)
        ]

        try:
            response = self.llm.invoke(messages)
            return {
                "answer": response.content,
                "sources": []
            }
        except Exception as e:
            print(f"Gemini API error: {str(e)}")
            return {
                "answer": f"Unable to generate an answer. API error: {str(e)}",
                "sources": []
            }

    def generate_response(
        self,
        query: str,
        context_docs: List[Document]
    ):
        context_texts = [
            doc.page_content for doc in context_docs
        ]
        context_text = "\n".join(context_texts)

        if self.chain:
            try:
                result = self.chain.invoke({
                    "question": query
                })

                answer = (
                    result.get("answer")
                    or result.get("output_text")
                    or ""
                )

                sources = []

                for doc in result.get("source_documents", []):
                    sources.append({
                        "text": doc.page_content,
                        "metadata": getattr(doc, "metadata", {})
                    })

                return {
                    "answer": answer,
                    "sources": sources
                }

            except Exception as e:
                print(f"Chain error: {str(e)}")

        print("Trying direct Gemini generation...")

        return self._fallback_generation(
            query,
            context_text
        )

    def set_retriever(self, retriever):
        self._create_chain(retriever)

    def reset_conversation(self):
        if self.memory:
            self.memory.clear()

    def get_conversation_history(self):
        if self.memory:
            return self.memory.chat_memory.messages

        return []
