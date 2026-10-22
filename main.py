from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from google import genai
import json
import os

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
    if not req.text or len(req.text.strip()) < 20:
        raise HTTPException(status_code=400, detail="El texto es muy corto.")
    
    prompt = f"""
    Eres un profesor experto creando exámenes. 
    Basándote en el siguiente texto de estudio, genera exactamente {req.num_questions} preguntas de opción múltiple.
    
    Reglas estrictas:
    1. Cada pregunta debe tener 4 opciones.
    2. Solo una opción debe ser correcta.
    3. Debes indicar el índice de la opción correcta (0, 1, 2 o 3).
    4. Añade una breve justificación de por qué es correcta.
    5. Las preguntas deben ser variadas y evaluar diferentes partes del texto.
    
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
    
    try:
        # Lista de modelos en orden de preferencia (del más capaz al más ligero)
        modelos = ['gemini-3.8-flash', 'gemini-3.5-flash', 'gemini-3.5-flash-lite']
        ultimo_error = None

        for modelo in modelos:
            try:
                print(f"Intentando con modelo: {modelo}...")
                response = client.models.generate_content(
                    model=modelo,
                    contents=prompt,
                )
                # Si llegamos aquí, funcionó. Procesamos la respuesta.
                texto_limpio = response.text.replace("```json", "").replace("```", "").strip()
                data = json.loads(texto_limpio)
                preguntas = data.get("questions", [])
                
                if preguntas:
                    print(f"✅ Éxito con {modelo}")
                    return GenerateResponse(questions=preguntas)
                else:
                    print(f"⚠️ {modelo} no devolvió preguntas válidas.")
                    
            except Exception as e:
                print(f"❌ Falló {modelo}: {e}")
                ultimo_error = e
                # Si el error es 503, intentamos con el siguiente modelo
                continue
        
        # Si ninguno funcionó, lanzamos el último error
        raise ultimo_error

    except Exception as e:
        print(f"--- ERROR FINAL DE GEMINI ---")
        print(f"Detalle: {e}")
        print(f"-----------------------------")
        raise HTTPException(status_code=500, detail=f"Error de IA: {str(e)}")