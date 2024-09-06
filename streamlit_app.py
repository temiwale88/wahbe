# Reference: modified using resource from shashankdeshpande: https://github.com/shashankdeshpande/langchain-chatbot/blob/master/pages/6_%F0%9F%94%97_chat_with_website.py
import os
import utils
import streamlit as st
from streaming import StreamHandler

from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.vectorstores import DocArrayInMemorySearch
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import Docx2txtLoader
from langchain_openai import OpenAIEmbeddings


st.set_page_config(page_title="ChatPDF", page_icon="📄")
st.header('Chat with the WAHBE RFP HBE 24-006')

class CustomDocChatbot:

    def __init__(self):
        utils.sync_st_session()
        self.llm = utils.configure_llm()
        self.embedding_model = utils.configure_embedding_model()

    def save_file(self, file):
        folder = 'tmp'
        if not os.path.exists(folder):
            os.makedirs(folder)
        
        file_path = f'./{folder}/{file.name}'
        with open(file_path, 'wb') as f:
            f.write(file.getvalue())
        return file_path

    def setup_qa_chain(self):
        # Load documents from the 'data' folder
        docs = []
        data_folder = 'data' 

        # Iterate through all .docx files in the data folder
        for filename in os.listdir(data_folder):
            if filename.endswith('.docx'):
                file_path = os.path.join(data_folder, filename)
                loader = Docx2txtLoader(file_path)
                docs.extend(loader.load())
        
        # Split documents and store in vector db
        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        splits = text_splitter.split_documents(docs)
        # vectordb = DocArrayInMemorySearch.from_documents(splits, self.embedding_model)
        vectordb = DocArrayInMemorySearch.from_documents(splits, OpenAIEmbeddings())

        # Define retriever
        retriever = vectordb.as_retriever(
            search_type='mmr',
            search_kwargs={'k': len(docs), 'fetch_k': 4}
        )


        # Setup memory for contextual conversation        
        memory = ConversationBufferMemory(
            memory_key='chat_history',
            output_key='answer',
            return_messages=True
        )

        # Setup LLM and QA chain
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=memory,
            return_source_documents=True,
            verbose=False
        )
        return qa_chain

    @utils.enable_chat_history
    def main(self):
        # User Inputs
        user_query = st.chat_input(placeholder="Ask me anything!")

        if not user_query:
            st.success("Enter your question below")
            st.stop()

        # Assuming the documents are already embedded and available
        qa_chain = self.setup_qa_chain()

        utils.display_msg(user_query, 'user')

        with st.chat_message("assistant"):
            st_cb = StreamHandler(st.empty())
            result = qa_chain.invoke(
                {"question": user_query},
                {"callbacks": [st_cb]}
            )
            # print("result: ", result)
            response = result["answer"]
            st.session_state.messages.append({"role": "assistant", "content": response})
            utils.print_qa(CustomDocChatbot, user_query, response)

            # to show references
            for idx, doc in enumerate(result['source_documents'], 1):
                filename = os.path.basename(doc.metadata['source'])
                # print(doc.metadata)
                # page_num = doc.metadata['page']
                ref_title = f":blue[Reference {idx}: *{filename}*]"
                with st.popover(ref_title):
                    st.caption(doc.page_content)


if __name__ == "__main__":
    obj = CustomDocChatbot()
    obj.main()
