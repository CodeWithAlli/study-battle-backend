# backend/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import spacy
import random
import re

# Cargar modelo de español
try:
    nlp = spacy.load("es_core_news_sm")
except OSError:
    raise RuntimeError(
        "Modelo 'es_core_news_sm' no encontrado. "
        "Ejecuta: python -m spacy download es_core_news_sm"
    )

app = FastAPI(title="Study Battle API", version="1.0.0")

# CORS: permite que el frontend (localhost:5173) llame al backend
# En producción, cambia esto por la URL de tu frontend en Vercel.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------- Modelos de datos ----------
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


# ---------- Generador de preguntas ----------
def generar_preguntas(texto: str, num: int) -> list[dict]:
    """
    Genera preguntas simples tipo "¿Quién hizo X?" o "¿Qué es Y?"
    usando análisis de dependencias de spaCy.
    Enfoque determinista: mismo texto → mismas preguntas.
    """
    doc = nlp(texto)
    preguntas = []

    for sent in doc.sents:
        # Buscar el verbo raíz de la oración
        root = sent.root
        if root.pos_ != "VERB":
            continue

        # Extraer sujeto y objeto directo
        sujeto = None
        objeto = None
        for child in root.children:
            if child.dep_ in ("nsubj", "nsubjpass"):
                sujeto = " ".join([t.text for t in child.subtree if not t.is_punct])
            elif child.dep_ in ("dobj", "obj", "attr"):
                objeto = " ".join([t.text for t in child.subtree if not t.is_punct])

        if not sujeto or not objeto:
            continue

        # Pregunta: ¿Qué hizo <sujeto>? (respuesta = verbo + objeto)
        verbo = root.lemma_
        pregunta_texto = f"¿Qué hizo {sujeto}?"
        respuesta_correcta = f"{verbo} {objeto}"
        justificacion = f"Según el texto, {sujeto} {verbo} {objeto}."

        # Generar distractores simples (otros verbos/objetos de la misma oración)
        distractores = []
        for token in sent:
            if token.pos_ == "VERB" and token.lemma_ != verbo and len(token.lemma_) > 2:
                distractores.append(f"{token.lemma_} algo")
            if token.pos_ == "NOUN" and token.text.lower() != objeto.lower() and len(token.text) > 3:
                distractores.append(f"{verbo} {token.text}")

        # Rellenar hasta 3 distractores únicos
        opciones = [respuesta_correcta]
        random.seed(42)  # determinista
        candidatos = list(set(distractores))
        random.shuffle(candidatos)

        for d in candidatos:
            if len(opciones) >= 4:
                break
            if d not in opciones:
                opciones.append(d)

        # Si no hay suficientes, rellenar con placeholders
        while len(opciones) < 4:
            opciones.append(f"Alternativa {len(opciones)}")

        # Mezclar opciones manteniendo el índice correcto
        random.shuffle(opciones)
        correct_index = opciones.index(respuesta_correcta)

        preguntas.append({
            "question_text": pregunta_texto,
            "options": opciones[:4],
            "correct_index": correct_index,
            "justification": justificacion,
        })

        if len(preguntas) >= num:
            break

    return preguntas


# ---------- Endpoints ----------
@app.get("/health")
def health():
    return {"status": "ok", "service": "study-battle-api"}


@app.post("/generate", response_model=GenerateResponse)
def generate(req: GenerateRequest):
    if not req.text or len(req.text.strip()) < 20:
        raise HTTPException(
            status_code=400,
            detail="El texto debe tener al menos 20 caracteres."
        )

    if req.num_questions < 1 or req.num_questions > 15:
        raise HTTPException(
            status_code=400,
            detail="num_questions debe estar entre 1 y 15."
        )

    preguntas = generar_preguntas(req.text, req.num_questions)

    if not preguntas:
        raise HTTPException(
            status_code=422,
            detail="No se pudieron generar preguntas. Intenta con un texto más descriptivo."
        )

    return GenerateResponse(questions=preguntas)