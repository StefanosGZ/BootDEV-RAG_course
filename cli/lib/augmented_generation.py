from dotenv import load_dotenv
from google import genai
import os

load_dotenv()
api_key = os.environ.get("GEMINI_API_KEY")
client = genai.Client(api_key=api_key)

def basic_rag(query, docs):
    RAG_RESPONSE = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"""You are a RAG agent for Hoopla, a movie streaming service.
                    Your task is to provide a natural-language answer to the user's query based on documents retrieved during search.
                    Provide a comprehensive answer that addresses the user's query.

                    Query: {query}

                    Documents:
                    {docs}

                    Answer:"""
    )
    return RAG_RESPONSE.text

def summarize_rag(query, results):
    RAG_RESPONSE = client.models.generate_content(
        model="gemini-2.5-flash-lite",
        contents=f"""Provide information useful to the query below by synthesizing data from multiple search results in detail.

                    The goal is to provide comprehensive information so that users know what their options are.
                    Your response should be information-dense and concise, with several key pieces of information about the genre, plot, etc. of each movie.

                    This should be tailored to Hoopla users. Hoopla is a movie streaming service.

                    Query: {query}

                    Search results:
                    {results}

                    Provide a comprehensive 3–4 sentence answer that combines information from multiple sources:"""
    )
    return RAG_RESPONSE.text

def citation_rag(query, documents):
    RAG_RESPONSE = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=f"""Answer the query below and give information based on the provided documents.

                The answer should be tailored to users of Hoopla, a movie streaming service.
                If not enough information is available to provide a good answer, say so, but give the best answer possible while citing the sources available.

                Query: {query}

                Documents:
                {documents}

                Instructions:
                - Provide a comprehensive answer that addresses the query
                - Cite sources in the format [1], [2], etc. when referencing information
                - If sources disagree, mention the different viewpoints
                - If the answer isn't in the provided documents, say "I don't have enough information"
                - Be direct and informative

                Answer:"""
    )
    return RAG_RESPONSE.text

def question_rag(question, context):
    RAG_RESPONSE = client.models.generate_content(
    model="gemini-2.5-flash-lite",
    contents=f"""Answer the user's question based on the provided movies that are available on Hoopla, a streaming service.

                Question: {question}

                Documents:
                {context}

                Instructions:
                - Answer questions directly and concisely
                - Be casual and conversational
                - Don't be cringe or hype-y
                - Talk like a normal person would in a chat conversation

                Answer:"""
    )
    return RAG_RESPONSE.text