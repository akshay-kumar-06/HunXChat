from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from openai import OpenAI
from elevenlabs.client import ElevenLabs
from langdetect import detect
import os
from dotenv import load_dotenv
import io
import traceback
# Load environment variables
load_dotenv()

app = FastAPI(title="HunXChat Voice AI API")

# CORS middleware - allows frontend to call backend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with your Vercel domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize clients
ELEVEN_API_KEY = os.getenv("ELEVEN_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCcl_fnRN3I0b_8jCuXUxVpX5J5mYoth0I")

eleven_client = ElevenLabs(api_key=ELEVEN_API_KEY)
openai_client = OpenAI(
    api_key=GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/"
)

# Persona context
PERSONA_CONTEXT = """
You are HunXChat, answering as Akshay.
 "name": "Akshay",
    "life_story": (
        "My story starts in a small village, where challenges were a part of life early on. That taught me one of my most important lessons: how to find a path through any difficulty. For me, that path was learning. I worked hard, got into a good school, and did well in my studies,but my real education was in learning how to overcome things."
        "When the pandemic hit, I turned inward. Exploring spirituality and meditation, and the wisdom of texts like the Mahabharat, gave me a deep sense of calm and a stronger belief in myself. I learned that true power isn't about being the loudest; it's about having that quiet confidence from within."
    ),
    "growth_areas": [
       "Building strong foundations in Machine Learning - understanding how today's smart systems actually work",
        "Mastering Generative AI to create new solutions and build powerful tools that help people",
        "Developing expertise in Prompt Engineering - because I've seen how the right questions lead to much better answers from AI systems",
        "Learning Cloud Computing - since this is where all modern applications live and scale to serve millions reliably"
    ],
    "superpower": (
         "When problems come my way, I don't see roadblocks - I see puzzles to solve. I take a moment to understand what's really happening, stay calm, and figure out how to move forward while learning something new."
    ),
    "misconception": (
        "People sometimes think that because I'm not the loudest voice in the room, I don't have ideas to share. The truth is, I'm listening, thinking, and when I speak, my ideas are clear and well-considered."
    ),
    "push_boundaries": (
       "I believe growth happens step by step. When I face a challenge, I learn what I can, talk to people who know more, think it through carefully, and then build my solution - always ready to learn and improve as I go."
    ),
    "brags": [
         "I learn quickly and can understand complex situations easily, which helps me contribute useful ideas to my team"
    ],
    please consider all above qualities specially technical,
    while answering consider: take the reference of above and refine it and then generate a refined version of answer,
    accent: indian, friendly, professional, happy with smile while answering, make sure it is concise as well.
"""

class TextRequest(BaseModel):
    text: str

def detect_lang(text: str) -> str:
    """Detect language of the text"""
    try:
        code = detect(text)
        if code.startswith("hi"):
            return "hi"
    except:
        pass
    return "en"

@app.get("/")
async def root():
    return {"message": "HunXChat Voice AI API is running", "status": "healthy"}

@app.post("/api/chat")
async def chat(request: TextRequest):
    """
    Process text input and return AI response
    """
    try:
        user_text = request.text
        
        # Detect language
        lang = detect_lang(user_text)
        
        # Generate response using Gemini
        prompt = f"{user_text}\nRespond in the user language."
        
        response = openai_client.chat.completions.create(
            model="gemini-2.5-flash",
            messages=[
                {'role': 'system', 'content': PERSONA_CONTEXT},
                {'role': 'user', 'content': prompt}
            ]
        )
        
        reply = response.choices[0].message.content.strip()
        
        return {
            "success": True,
            "text": reply,
            "language": lang
        }
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/synthesize")
async def synthesize_speech(request: TextRequest):
    """
    Convert text to speech using ElevenLabs
    """
    try:
        text = request.text
        
        # Generate audio
        audio_generator = eleven_client.text_to_speech.convert(
            text=text,
            voice_id="21m00Tcm4TlvDq8ikWAM",
            model_id="eleven_turbo_v2_5"
        )
        
        # Collect audio chunks
        audio_bytes = b"".join(audio_generator)
        
        # Return audio as streaming response
        return StreamingResponse(
            io.BytesIO(audio_bytes),
            media_type="audio/mpeg",
            headers={
                "Content-Disposition": "inline; filename=speech.mp3"
            }
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/voice-chat")
async def voice_chat(audio: UploadFile = File(...)):
    """
    Complete voice chat pipeline: audio input -> text -> AI response -> audio output
    """
    try:
        # For now, we'll rely on frontend to handle speech-to-text
        # This endpoint can be extended to handle audio transcription if needed
        return {
            "success": True,
            "message": "Use /api/chat and /api/synthesize for voice pipeline"
        }
    
    except Exception as e:
        print("--- !!! CRITICAL ERROR IN /api/synthesize !!! ---")
        print(traceback.format_exc()) # This prints the full error
        print("--- !!! END OF ERROR REPORT !!! ---")
        raise HTTPException(status_code=500, detail=str(e))



