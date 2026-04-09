import os
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from supabase import create_client, Client
from google import genai
from google.genai import types

load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
ai_client = genai.Client(api_key=GEMINI_API_KEY)

app = FastAPI(title="SoilPatrol Multi-Tool AI Brain")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

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
        search_query = f"Crops that thrive in pH {request.sensorData.phLevel}, Temp {request.sensorData.airTempC}C, and Moisture {request.sensorData.soilMoisturePct}%"
        context = search_crop_knowledge(search_query)

        prompt = f"""
        TASK: Suggest 3 optimal crops based on this live sensor data:
        N:{request.sensorData.nitrogenMg}, P:{request.sensorData.phosphorusMg}, K:{request.sensorData.potassiumMg}, pH:{request.sensorData.phLevel}, Temp:{request.sensorData.airTempC}C.
        
        KNOWLEDGE BASE: {context}
        
        REQUIREMENTS:
        1. Current Soil Stats summary.
        2. Suggest 3 crops. For each, list ideal conditions, required soil adjustments (based on EXPERT RULES), expected yield, and market price potential.
        """
        response = ai_client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config=types.GenerateContentConfig(system_instruction=EXPERT_RULES, temperature=0.3))
        return {"status": "success", "report": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# TOOL 2: SOIL CHECK-IN
# ==========================================
@app.post("/api/ai/check-in")
async def tool_soil_check_in(request: CheckInRequest):
    try:
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
        response = ai_client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config=types.GenerateContentConfig(system_instruction=EXPERT_RULES, temperature=0.2))
        return {"status": "success", "report": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# ==========================================
# TOOL 3: CROP SUITABILITY
# ==========================================
@app.post("/api/ai/suitability")
async def tool_crop_suitability(request: SuitabilityRequest):
    try:
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
        response = ai_client.models.generate_content(model="gemini-2.5-flash", contents=prompt, config=types.GenerateContentConfig(system_instruction=EXPERT_RULES, temperature=0.2))
        return {"status": "success", "report": response.text}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))