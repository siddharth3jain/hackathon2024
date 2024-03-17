from initialization import llm
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor
from langchain.chains import RetrievalQA
from chunking import vectorstore


class ESG_runner:
    def _init_(self):
        self.response = 'documents'

    # input pdf documents
    def Input_docs(self, documents_full_path):
        self.vectordb = vectorstore(documents_full_path)
        pdf_docs = [i['source'].split('\\')[-1] for i in self.vectordb.get()['metadatas']]
        return list(dict.fromkeys(pdf_docs))

    # get answer from LLM
    def llm_call(self, question):
        # Contextual compression
        compressor = LLMChainExtractor.from_llm(llm)
        compression_retriever = ContextualCompressionRetriever(
            base_compressor=compressor,
            base_retriever=self.vectordb.as_retriever()
        )

        # Retrival qa
        qa_chain = RetrievalQA.from_chain_type(
            llm,
            retriever=compression_retriever,
            return_source_documents=True,
            chain_type="map_reduce"
        )

        result = qa_chain({"query": question})
        return result

    # PDF questions
    def pdf_questions(self, questions_folder_path):
        # self.pdf_questions = pdfparser(questions_folder_path)
        # response_dict = dict()
        # for question in self.pdf_questions:
        #     response_dict[question] = self.llm_call(question)

        # return response_dict
        return "Its still under development"

    # Prompt questions
    def prompt_question(self, question):
        return self.llm_call(question)