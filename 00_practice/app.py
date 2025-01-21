import streamlit as st
from langchain_groq.chat_models import ChatGroq
from langchain.schema import HumanMessage, SystemMessage
import requests
import json
from dotenv import load_dotenv

load_dotenv()


# Initialize the ChatGroq LLM
llm = ChatGroq(model='llama-3.2-90b-vision-preview')

# Function to classify the user's query using the LLM
def classify_query(query: str) -> str:
    """
    Classifies the user's query to determine which API endpoint to fetch data from.
    
    Args:
        query (str): The user's input query.
    
    Returns:
        str: The API endpoint to fetch data from.
    """
    # Define the system message to guide the LLM
    system_message = SystemMessage(content="""
    You are a helpful assistant that classifies user queries into one of the following categories:
    - products: If the query is related to products, items, or goods.
    - pokemon: If the query is related to Pokémon, creatures, or games.
    - users: If the query is related to users, people, or profiles.
    - movies: If the query is related to movies, films, or IMDB ratings.
    
    Return only the category name (e.g., "products", "pokemon", etc.) based on the user's query.
    """)

    # Create a human message with the user's query
    human_message = HumanMessage(content=query)

    # Generate a response using the LLM
    ai_message = llm.invoke([system_message, human_message])

    # Extract the category from the LLM's response
    category = ai_message.content.strip().lower()

    # Validate the category
    valid_categories = ["products", "pokemon", "users", "movies"]
    if category in valid_categories:
        return category
    else:
        return None

# Function to fetch products data from the Dummy API
def fetch_products():
    """
    Fetches products data from Dummy API.
    
    Returns:
        dict: A dictionary containing the products data.
    """
    response = requests.get('https://dummyapi.online/api/products')
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        return None

# Function to fetch pokemon data from the Dummy API
def fetch_pokemon():
    """
    Fetches pokemon data from Dummy API.
    
    Returns:
        dict: A dictionary containing the pokemon data.
    """
    response = requests.get('https://dummyapi.online/api/pokemon')
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        return None

# Function to fetch users data from the Dummy API
def fetch_users():
    """
    Fetches users data from Dummy API.
    
    Returns:
        dict: A dictionary containing the users data.
    """
    response = requests.get('https://dummyapi.online/api/users')
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        return None

# Function to fetch movies data from the Dummy API
def fetch_movies():
    """
    Fetches 100 movies with IMDB ratings from Dummy API.
    
    Returns:
        dict: A dictionary containing the movies data.
    """
    response = requests.get('https://dummyapi.online/api/movies')
    if response.status_code == 200:
        return json.loads(response.text)
    else:
        return None

# Function to fetch data from the appropriate API
def fetch_data(api_endpoint: str):
    """
    Fetches data from the specified Dummy API endpoint.
    
    Args:
        api_endpoint (str): The API endpoint to fetch data from.
    
    Returns:
        dict: A dictionary containing the fetched data.
    """
    if api_endpoint == "products":
        return fetch_products()
    elif api_endpoint == "pokemon":
        return fetch_pokemon()
    elif api_endpoint == "users":
        return fetch_users()
    elif api_endpoint == "movies":
        return fetch_movies()
    else:
        return None

# Function to process the user's prompt
def process_prompt(prompt: str, data: dict):
    """
    Processes the user's prompt using the fetched data.
    
    Args:
        prompt (str): The user's input prompt.
        data (dict): The fetched data from the API.
    
    Returns:
        str: The AI's response to the user's prompt.
    """
    # Create a system message with the fetched data
    system_message = SystemMessage(content=f"Data: {data}")

    # Create a human message with the user's prompt
    human_message = HumanMessage(content=prompt)

    # Generate a response using the LLM
    ai_message = llm.invoke([system_message, human_message])

    return ai_message.content

# Streamlit UI
def main():
    st.title("API Data Query Assistant")
    st.write("Ask a question related to products, Pokémon, users, or movies.")

    # Get user input
    user_input = st.text_input("Enter your query:")

    if user_input:
        # Classify the query to determine which API to fetch data from
        api_endpoint = classify_query(user_input)
        
        if api_endpoint:
            # Fetch data from the appropriate API
            data = fetch_data(api_endpoint)
            
            if data:
                # Process the prompt using the fetched data
                response_content = process_prompt(user_input, data)
                
                # Display the AI's response
                st.write(f"**Assistant:** {response_content}")
            else:
                st.error(f"Failed to fetch data from the {api_endpoint} API.")
        else:
            st.warning("Sorry, I couldn't determine which data to fetch based on your query.")

if __name__ == "__main__":
    main()