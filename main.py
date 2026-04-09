import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from supabase import create_client, Client
from google import genai
from google.genai import types
from cachetools import TTLCache


load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="SoilPatrol Multi-Tool AI Brain")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])
report_cache = TTLCache(maxsize=1000, ttl=10800)

def generate_cache_key(tool_name: str, request_dict: dict) -> str:
    """Creates a unique ID for this specific request to check if we've seen it recently."""
    # Convert dict to a string and hash it
    dict_str = json.dumps(request_dict, sort_keys=True)
    return hashlib.md5(f"{tool_name}_{dict_str}".encode()).hexdigest()
# --- DATA MODELS ---
class SensorData(BaseModel):
    airTempC: float
    airHumidityPct: float
    soilMoisturePct: float
    soilTempC: float
    phLevel: float
    nitrogenMg: float
    phosphorusMg: float
    potassiumMg: float

class CheckInRequest(BaseModel):
    sensorData: SensorData
    currentCrop: str

class SuitabilityRequest(BaseModel):
    sensorData: SensorData
    targetCrop: str

class SuggestionRequest(BaseModel):
    sensorData: SensorData

# --- THE HARDCODED EXPERT RULES (From your CSVs) ---
EXPERT_RULES = """
You are the SoilPatrol Master Agronomist. Follow these absolute rules derived from our agronomy database:
1. CRITICAL LIMITS: N < 40mg/kg is Low. P < 15mg/kg is Low. K < 100mg/kg is Low.
2. FERTILIZER CORRECTION: 
   - If N is Low: Recommend 100 kg/ha Urea broadcast.
   - If P is Low: Recommend 50 kg/ha DAP base application.
   - If K is Low: Recommend 40 kg/ha MOP.
3. pH LOGIC:
   - If pH < 5.5: Recommend Agricultural Lime.
   - If pH > 7.5: Recommend Elemental Sulfur.
4. FORMAT: Always use clear Markdown with bullet points. Include Yield Expectations and Price Estimates where relevant.
"""

# Helper function to query the Vector Database
def search_crop_knowledge(query: str):
    response = ai_client.models.embed_content(model="gemini-embedding-001", contents=query)
    query_vector = response.embeddings[0].values
    db_response = supabase.rpc("match_crops", {"query_embedding": query_vector, "match_threshold": 0.4, "match_count": 3}).execute()
    return "\n".join([item['content'] for item in db_response.data])

# ==========================================
# TOOL 1: SUGGEST ME A CROP
# ==========================================
@app.post("/api/ai/suggest")
async def tool_suggest_crop(request: SuggestionRequest):
    try:
        # 1. Check the Memory first!
        cache_key = generate_cache_key("suggest", request.model_dump())
        if cache_key in report_cache:
            print("CACHE HIT! Returning saved report.")
            return {"status": "success", "report": report_cache[cache_key], "cached": True}

        print("CACHE MISS. Generating new report...")
        # 2. Generate the report (your existing logic)
        search_query = f"Crops that thrive in pH {request.sensorData.phLevel}, Temp {request.sensorData.airTempC}C, and Moisture {request.sensorData.soilMoisturePct}%"
        context = search_crop_knowledge(search_query)

        prompt = f"""... [Keep your existing prompt here] ..."""
        
        response = ai_client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt, 
            config=types.GenerateContentConfig(system_instruction=EXPERT_RULES, temperature=0.3)
        )
        
        # 3. Save the new report to Memory for the next 3 hours
        report_cache[cache_key] = response.text
        
        return {"status": "success", "report": response.text, "cached": False}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# TOOL 2: SOIL CHECK-IN (WITH CACHE)
# ==========================================
@app.post("/api/ai/check-in")
async def tool_soil_check_in(request: CheckInRequest):
    try:
        # 1. Check the Memory first!
        cache_key = generate_cache_key("check_in", request.model_dump())
        if cache_key in report_cache:
            print("CACHE HIT! Returning saved check-in report.")
            return {"status": "success", "report": report_cache[cache_key], "cached": True}

        print("CACHE MISS. Generating new check-in report...")
        # 2. Generate the report
        context = search_crop_knowledge(f"Ideal parameters for {request.currentCrop}")
        
        prompt = f"""
        TASK: Perform a Soil Check-In for the actively growing crop: {request.currentCrop}.
        LIVE DATA: N:{request.sensorData.nitrogenMg}, P:{request.sensorData.phosphorusMg}, K:{request.sensorData.potassiumMg}, pH:{request.sensorData.phLevel}.
        
        KNOWLEDGE BASE: {context}
        
        REQUIREMENTS:
        1. Compare Live Stats vs Ideal Stats for {request.currentCrop}.
        2. Identify deficiencies.
        3. Provide actionable fertilizer/amendment recommendations based on the EXPERT RULES.
        4. If the soil is catastrophically unsuited for {request.currentCrop}, suggest an alternative.
        """
        response = ai_client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt, 
            config=types.GenerateContentConfig(system_instruction=EXPERT_RULES, temperature=0.2)
        )
        
        # 3. Save the new report to Memory for the next 3 hours
        report_cache[cache_key] = response.text
        
        return {"status": "success", "report": response.text, "cached": False}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# TOOL 3: CROP SUITABILITY (WITH CACHE)
# ==========================================
@app.post("/api/ai/suitability")
async def tool_crop_suitability(request: SuitabilityRequest):
    try:
        # 1. Check the Memory first!
        cache_key = generate_cache_key("suitability", request.model_dump())
        if cache_key in report_cache:
            print("CACHE HIT! Returning saved suitability report.")
            return {"status": "success", "report": report_cache[cache_key], "cached": True}

        print("CACHE MISS. Generating new suitability report...")
        # 2. Generate the report
        context = search_crop_knowledge(f"Ideal parameters and environmental requirements for {request.targetCrop}")
        
        prompt = f"""
        TASK: Evaluate if the farmer should plant {request.targetCrop} in this soil.
        LIVE DATA: N:{request.sensorData.nitrogenMg}, P:{request.sensorData.phosphorusMg}, K:{request.sensorData.potassiumMg}, pH:{request.sensorData.phLevel}, Temp:{request.sensorData.airTempC}C.
        
        KNOWLEDGE BASE: {context}
        
        REQUIREMENTS:
        1. Provide a definitive GO / NO-GO decision for planting {request.targetCrop}.
        2. Show the delta between the Live Data and the Ideal Stats for this crop.
        3. List the required interventions (Fertilizer/Lime) needed before planting can begin.
        """
        response = ai_client.models.generate_content(
            model="gemini-1.5-flash", 
            contents=prompt, 
            config=types.GenerateContentConfig(system_instruction=EXPERT_RULES, temperature=0.2)
        )
        
        # 3. Save the new report to Memory for the next 3 hours
        report_cache[cache_key] = response.text
        
        return {"status": "success", "report": response.text, "cached": False}
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        raise HTTPException(status_code=500, detail=str(e))