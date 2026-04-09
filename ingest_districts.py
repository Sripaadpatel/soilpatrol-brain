import pandas as pd
import os
from dotenv import load_dotenv
from google import genai
from supabase import create_client, Client

load_dotenv()

supabase: Client = create_client(os.getenv("SUPABASE_URL"), os.getenv("SUPABASE_KEY"))
ai_client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

print("Loading District Data...")
df = pd.read_csv("data/cleaned_District-wise_Key_Data.csv") 

# FIX 1: Clean up any invisible spaces in the column headers
df.columns = df.columns.str.strip()

# FIX 2: Print the columns so we can see exactly what they are named
print("Found these columns in your CSV:", df.columns.tolist())

for index, row in df.iterrows():
    district = row['District']
    
    # Notice I used .get() here. If a column is missing, it will safely return "Unknown" instead of crashing!
    content = (
        f"District: {district}, State: {row.get('State', 'Telangana')}. "
        f"Major Soil Types: {row.get('Major Soil Types', row.get('Major Soil Type', 'Unknown'))}. "
        f"Average Nitrogen: {row.get('Avg N (mg/kg)', 'Unknown')} mg/kg. "
        f"Average Phosphorus: {row.get('Avg P (mg/kg)', 'Unknown')} mg/kg. "
        f"Average Potassium: {row.get('Avg K (mg/kg)', 'Unknown')} mg/kg. "
        f"Typical pH: {row.get('Avg pH', 'Unknown')}. "
        f"Dominant Crops: {row.get('Dominant Crops', 'Unknown')}."
    )
    
    print(f"Embedding data for {district}...")
    response = ai_client.models.embed_content(model="gemini-embedding-001", contents=content)
    embedding_vector = response.embeddings[0].values
    
    supabase.table('district_knowledge').insert({
        "district_name": district,
        "content": content,
        "embedding": embedding_vector
    }).execute()

print("District knowledge successfully loaded!")