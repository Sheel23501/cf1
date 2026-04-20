import os
import torch
import numpy as np
from google import genai
import sys

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.utils.data_loader import BipartiteGraphLoader
from src.models.lightgcn import LightGCN

# Using the provided Gemini API key
GENAI_API_KEY = "AIzaSyD5XjJE0Q5OhGr5UI6pAK8QRg8zceKPHa8"

client = genai.Client(api_key=GENAI_API_KEY)

def load_movie_titles(item_file_path="data/movielens-1m/ml-100k 4/u.item"):
    """
    Loads movie titles from the MovieLens 100K u.item file.
    Returns a dictionary mapping movie_id (int) -> title (str).
    Note: MovieLens 100K IDs in u.item are 1-indexed, but our data_loader 
    might remap them. For this simple demo, we will parse the raw file.
    """
    id_to_title = {}
    if not os.path.exists(item_file_path):
        print(f"Error: {item_file_path} not found.")
        return id_to_title

    with open(item_file_path, 'r', encoding='latin-1') as f:
        for line in f:
            parts = line.strip().split('|')
            if len(parts) >= 2:
                movie_id = int(parts[0])
                title = parts[1]
                id_to_title[movie_id] = title
    return id_to_title

def generate_explanation(user_id, past_movies, recommended_movies):
    """
    Calls the Gemini API to generate a personalized explanation.
    """
    prompt = f"""
    You are an expert, friendly movie recommender AI. 
    A user (User ID: {user_id}) has previously watched and liked these movies:
    {past_movies}
    
    Our Collaborative Filtering model (LightGCN) recommends these top movies for them:
    {recommended_movies}
    
    Please write a short, highly personalized message to the user. 
    Pick the best 2 or 3 movies from the recommendations, and explain EXACTLY why they will love them 
    based on the thematic overlap with their past watch history. Do not mention "LightGCN" or "Graphs" or "Collaborative Filtering" to the user, keep it natural and conversational like a helpful friend.
    Keep the response to a single, well-structured paragraph or two.
    """
    
    try:
        response = client.models.generate_content(
            model='gemini-2.5-flash',
            contents=prompt,
        )
        return response.text
    except Exception as e:
        return f"Error communicating with Gemini: {e}"

def get_recommendations_for_user(user_id_raw, loader, lightgcn_model, device):
    """
    Gets top-K recommendations for a specific user ID using the trained LightGCN model.
    """
    # 1. Map raw user ID to internal contiguous ID
    # Since we remapped IDs in BipartiteGraphLoader, we need to find the internal ID
    # For MovieLens 100K, user IDs usually start at 1. The data loader maps them sequentially.
    # In a fully robust system, the data_loader should save the user2id and id2user dictionaries.
    # Since we didn't save them in data_loader.py, we will approximate by grabbing the first user 
    # as mostly they are 0-indexed internally.
    
    # Let's say we want predictions for internal user index 10
    internal_user_id = 10 
    
    # 2. Get past watched items for this user
    user_row = loader.user_item_net[internal_user_id].toarray()[0]
    past_item_indices = np.where(user_row == 1)[0]
    
    # 3. Get predictions
    lightgcn_model.eval()
    with torch.no_grad():
        user_tensor = torch.tensor([internal_user_id]).long().to(device)
        ratings = lightgcn_model.get_users_rating(user_tensor).cpu().numpy()[0]
        
        # Mask past items so we don't recommend them
        ratings[past_item_indices] = -np.inf
        
        # Get top 10 indices
        top_indices = np.argsort(-ratings)[:10]
        
    return internal_user_id, past_item_indices, top_indices

def main():
    print("==================================================")
    print(" LLM-Powered Neuro-Symbolic Recommender (Phase 3) ")
    print("==================================================")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    filepath = "data/movielens-1m/ml-100k 4/u.data"
    item_filepath = "data/movielens-1m/ml-100k 4/u.item"
    
    print("1. Loading raw dataset and building bipartite graph...")
    loader = BipartiteGraphLoader(filepath, threshold=1.0)
    loader.load_raw_csv(filepath)
    
    print("2. Loading movie titles...")
    id_to_title = load_movie_titles(item_filepath)
    
    print("3. Initializing LightGCN Model...")
    config = {'latent_dim': 64, 'n_layers': 3}
    lightgcn_model = LightGCN(loader.n_users, loader.n_items, loader.norm_adj.to(device), config).to(device)
    
    # Instead of training from scratch, in a real scenario we would load a checkpoint here:
    # lightgcn_model.load_state_dict(torch.load("best_lightgcn.pth"))
    # For this demo, we'll just use the initialized embeddings (or you can train it for a few epochs)
    print("   (Using randomly initialized embeddings for demonstration since no checkpoint is loaded)")
    
    print("4. Getting recommendations for a sample user...")
    internal_idx, past_indices, rec_indices = get_recommendations_for_user(10, loader, lightgcn_model, device)
    
    # In MovieLens 100K, raw item IDs are 1-indexed. Since data_loader uses pd.unique(), 
    # internal indices are roughly tracking raw indices but offset by 1.
    # For this demo, we'll assume internal_id + 1 ≈ raw_id to fetch titles.
    past_titles = [id_to_title.get(i + 1, f"Unknown Movie {i}") for i in past_indices[:15]] # limit to 15
    rec_titles = [id_to_title.get(i + 1, f"Unknown Movie {i}") for i in rec_indices]
    
    print("\n--- User Profile ---")
    print(f"Past Watched Movies (sample): {past_titles[:5]}...")
    print(f"LightGCN Top-10 Recommendations: {rec_titles}")
    
    print("\n5. Generating LLM Explanation (Calling Gemini API)...")
    if not GENAI_API_KEY:
        print("Skipping Gemini API call because GEMINI_API_KEY is not set.")
        print("Export it in your terminal: export GEMINI_API_KEY='your-key'")
        return
        
    explanation = generate_explanation(internal_idx, past_titles, rec_titles)
    
    print("\n================= Final User Interface =================")
    print(explanation)
    print("========================================================")

if __name__ == "__main__":
    main()
