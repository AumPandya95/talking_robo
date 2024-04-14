import nest_asyncio
import pandas as pd
from typing import Optional
from dataclasses import dataclass
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import AsyncChromiumLoader
from langchain.document_transformers import Html2TextTransformer
from langchain.vectorstores import FAISS


nest_asyncio.apply()


@dataclass
class GenerateEmbeddings:
    """
    Class to generate embeddings for vector database
    """

    model: str
    articles: Optional[list] = None
    data_to_embed: Optional[pd.DataFrame] = None

    def router(self):
        """Decide next steps based on user input"""
        if self.articles:
            try:
                if not self.data_to_embed.empty:
                    # combine articles and data into one embedding
                    documents_from_articles = self.articles_to_documents()
                    documents_from_data = self.data_to_documents()
            except AttributeError:
                # extract documents from links only
                documents_from_articles = self.articles_to_documents()
        else:
            try:
                if not self.data_to_embed.empty:
                    # Data to document
                    documents_from_data = self.data_to_documents()
            except AttributeError:
                ### No data present for storing into vector DB
                pass
        

    def articles_to_documents(self):
        """Extract documents from websites"""
        # Scrapes the blogs above
        article_loader = AsyncChromiumLoader(self.articles)
        documents = article_loader.load()
        # Converts HTML to plain text
        html2text = Html2TextTransformer()
        transformed_documents = html2text.transform_documents(documents)

        return transformed_documents

    def data_to_documents(self):
        """
        Extract documents from structured data; data output from Stack 
        exchange query builder
        """
        pass




# Generate documents from articles
# Generate documents from data
# Chunk the data
# Use sentence transformer model to emed the data
# Use FAISS to store the data
# Pickle the data
