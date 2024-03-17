# The code snippet you provided is setting up a Gradio interface for a question-answering system. Here's a breakdown of what each part of the code is doing:
from pathlib import Path
from parameters import chroma_persist_directory
from langchain_community.vectorstores import Chroma
from initialization import embeddings_function
from langchain.chains.question_answering import load_qa_chain
from initialization import llm
import gradio as gr
from fastapi import FastAPI


app = FastAPI()

# load from disk
# Load all pdf vectorstore
# The line `db = Chroma(persist_directory=str(Path(chroma_persist_directory,'combined')), embedding_function=embeddings_function)` is creating an instance of the `Chroma` class with specific parameters. Here's a breakdown of what it's doing:
db = Chroma(persist_directory=str(Path(chroma_persist_directory, 'combined')), embedding_function=embeddings_function)


def qa_response(files, query):
    docs = db.similarity_search(query)
    # print results
    # print(docs[0].page_content)
    print(files)
    # Run the chain by passing the output of the similarity search
    chain = load_qa_chain(llm=llm, chain_type="stuff")
    res = chain({"input_documents": docs, "question": query})
    return res["output_text"]


demo = gr.Interface(fn=qa_response,
                    inputs=[gr.File(file_count='multiple'), "text"],
                    outputs=[gr.Textbox(label="Asnwer")]
                    )


app = gr.mount_gradio_app(app, demo, path="/")
# demo.launch()
