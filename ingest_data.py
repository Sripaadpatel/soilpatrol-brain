

import pandas as pd
from google import genai
from supabase import create_client, Client

# --- 1. SETUP YOUR KEYS HERE ---
SUPABASE_URL = "https://sdbjudcvufxxqdhuzdyj.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InNkYmp1ZGN2dWZ4eHFkaHV6ZHlqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc3NTQ2MjM2NCwiZXhwIjoyMDkxMDM4MzY0fQ.46STSqdxBCFL-ky8vkqjxi3t9kYEYrB6uj_HicravkA" # Use the service role key for inserting
# Get this from Google AI Studio
GEMINI_API_KEY = "AIzaSyATJGZyQr2ID4ELwOw_83-oOTx-3nKAiP8"
# Initialize Clients
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
# NEW: Google's updated client initialization
ai_client = genai.Client(api_key=GEMINI_API_KEY)

# --- 2. LOAD YOUR PERFECT CSV ---
print("Loading Crop Data...")
# Make sure this path points to your actual CSV location
df = pd.read_csv("data/cleaned_Crop_Ideal_Parameters.csv") 

# --- 3. PROCESS AND EMBED ---
for index, row in df.iterrows():
    crop = row['Crop']
    
    content = (
        f"Crop: {crop}. Category: {row['Category']}. "
        f"Ideal Nitrogen (N): {row['N Min (kg/ha)']} to {row['N Max (kg/ha)']} kg/ha. "
        f"Ideal Phosphorus (P): {row['P Min (kg/ha)']} to {row['P Max (kg/ha)']} kg/ha. "
        f"Ideal Potassium (K): {row['K Min (kg/ha)']} to {row['K Max (kg/ha)']} kg/ha. "
        f"Ideal pH: {row['pH Min']} to {row['pH Max']}. "
        f"Optimal Temperature: {row['Air Temp Min (°C)']}°C to {row['Air Temp Max (°C)']}°C. "
        f"Ideal Soil Moisture: {row['Soil Moisture Min (% VWC)']} to {row['Soil Moisture Max (% VWC)']}% VWC. "
        f"Notes: {row['Notes']}"
    )
    
    print(f"Generating Gemini Vector for {crop}...")
    
    # NEW: Google's updated embedding syntax
    response = ai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=content
    )
    
    # Extract the vector array
    embedding_vector = response.embeddings[0].values
    
    # Save to Supabase
    data, count = supabase.table('crop_knowledge').insert({
        "crop_name": crop,
        "content": content,
        "embedding": embedding_vector
    }).execute()
    
    print(f"Successfully saved {crop} to AI Brain!")

print("All crop knowledge ingested perfectly. Ready for RAG!")