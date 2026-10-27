from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
from google.genai import types
from dotenv import load_dotenv
import json
import os
import base64
import time
from typing import List, Optional

load_dotenv()

# Configurar Gemini con la nueva librería
API_KEY = os.getenv("GEMINI_API_KEY")
client = genai.Client(api_key=API_KEY)

app = FastAPI(title="Study Battle API", version="2.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

class GenerateRequest(BaseModel):
    text: str
    num_questions: int = 5
    images: List[str] = []

class QuestionOut(BaseModel):
    question_text: str
    options: list[str]
    correct_index: int
    justification: str

class GenerateResponse(BaseModel):
    questions: list[QuestionOut]

@app.get("/health")
def health():
    return {"status": "ok", "service": "study-battle-api-gemini"}

@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.text or (len(req.text.strip()) < 20 and not req.images):
        raise HTTPException(status_code=400, detail="El texto es muy corto y no hay imágenes.")

    prompt = f"""
    Eres un profesor experto creando exámenes. 
    Basándote en el siguiente texto de estudio Y EN LAS IMÁGENES O DOCUMENTOS ADJUNTOS, genera exactamente {req.num_questions} preguntas de opción múltiple.
    
    Si hay imágenes de gráficos, diagramas o dispositivos, úsalas para hacer preguntas visuales.
    
    Devuelve ÚNICAMENTE un objeto JSON válido con esta estructura exacta, sin markdown, sin texto extra:
    {{
      "questions": [
        {{
          "question_text": "¿Pregunta aquí?",
          "options": ["Opción A", "Opción B", "Opción C", "Opción D"],
          "correct_index": 0,
          "justification": "Justificación aquí."
        }}
      ]
    }}
    
    Texto de estudio:
    {req.text}
    """

    # Preparamos el contenido para Gemini (texto + imágenes/PDFs)
    contents = [prompt]
    
    for img_b64 in req.images:
        try:
            header, encoded = img_b64.split(",", 1)
            mime_type = header.split(";")[0].split(":")[1]
            image_bytes = base64.b64decode(encoded)
            
            contents.append(
                types.Part.from_bytes(
                    data=image_bytes,
                    mime_type=mime_type
                )
            )
        except Exception as e:
            print(f"Error decodificando archivo adjunto: {e}")
            continue

    # 🔥 SISTEMA DE RESPALDO: Prueba con varios modelos si uno falla por saturación (503)
    modelos = [
        'gemini-3.5-flash-lite',   # El más ligero y rápido
        'gemini-3.8-flash',        # El más potente
        'gemini-2.5-flash'         # Respaldo antiguo
    ]
    
    ultimo_error = None

    for modelo in modelos:
        try:
            print(f"Intentando generar con: {modelo}...")
            response = client.models.generate_content(
                model=modelo,
                contents=contents,
            )
            
            texto_limpio = response.text.replace("```json", "").replace("```", "").strip()
            data = json.loads(texto_limpio)
            preguntas = data.get("questions", [])
            
            if not preguntas:
                raise ValueError("La IA no devolvió preguntas válidas")
                
            print(f"✅ Éxito con {modelo}")
            return GenerateResponse(questions=preguntas)
            
        except Exception as e:
            print(f"❌ Falló {modelo}: {e}")
            ultimo_error = e
            # Esperamos 1 segundo antes de probar el siguiente modelo
            time.sleep(1) 
            continue 

    # Si ningún modelo funcionó
    print(f"--- ERROR FINAL: TODOS LOS MODELOS FALLARON ---")
    print(f"Último detalle: {ultimo_error}")
    raise HTTPException(
        status_code=500, 
        detail=f"Todos los modelos de IA están saturados en este momento. Intenta de nuevo en unos minutos. (Error: {str(ultimo_error)})"
    )