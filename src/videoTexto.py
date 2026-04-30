import moviepy as mp
from faster_whisper import WhisperModel
from deep_translator import GoogleTranslator
import json

# ===================================
# CONFIGURACION
# ===================================
properties = {
    "VIDEO_FILE": "uno.mp4",
    "AUDIO_FILE": "audio.wav",
    "MODEL_SIZE": "medium",
    "DEVICE": "cpu",
    "ARCHIVO_TRANSCRITO": "transcripcion_es.txt",
    "TRANSCRIPCION" : "transcripcion.txt"
}

# ===================================
# 1️⃣ EXTRAER AUDIO
# ===================================
print("Extrayendo audio...")

video = mp.VideoFileClip(properties["VIDEO_FILE"])
video.audio.write_audiofile(
    properties["AUDIO_FILE"],
    fps=16000,
    codec="pcm_s16le"
)

# ===================================
# 2️⃣ CARGAR WHISPER
# ===================================
print("Cargando modelo Whisper...")

model = WhisperModel(
    properties["MODEL_SIZE"],
    device=properties["DEVICE"],
    compute_type="int8"
)

# ===================================
# 3️⃣ TRANSCRIPCION
# ===================================
print("Transcribiendo...")

segments, info = model.transcribe(
    properties["AUDIO_FILE"],
    beam_size=5,
    task="transcribe"   # ← mantiene idioma original
)

print(f"Idioma detectado: {info.language}")

texto_total = ""
srt_content = ""
json_segments = []

# ===================================
# 4️⃣ PROCESAR RESULTADOS
# ===================================
for i, segment in enumerate(segments, start=1):

    texto_total += segment.text + " "

    def format_time(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    srt_content += f"{i}\n"
    srt_content += f"{format_time(segment.start)} --> {format_time(segment.end)}\n"
    srt_content += f"{segment.text.strip()}\n\n"

    json_segments.append({
        "start": segment.start,
        "end": segment.end,
        "text": segment.text.strip()
    })

# ===================================
# 5️⃣ GUARDAR TRANSCRIPCION ORIGINAL
# ===================================
print("Guardando transcripción...")

#with open("transcripcion.txt", "w", encoding="utf-8") as f:
with open(properties["ARCHIVO_TRANSCRITO"], "w", encoding="utf-8") as f:
    f.write(texto_total)

with open("subtitulos.srt", "w", encoding="utf-8") as f:
    f.write(srt_content)

with open("transcripcion.json", "w", encoding="utf-8") as f:
    json.dump(json_segments, f, indent=2, ensure_ascii=False)

# ===================================
# 6️⃣ TRADUCCION A ESPAÑOL ⭐ NUEVO PASO
# ===================================
print("Traduciendo a español...")

translator = GoogleTranslator(source="auto", target="es")

# dividir texto largo (evita límite Google)
def traducir_texto_largo(texto, chunk_size=4000):
    partes = [
        texto[i:i+chunk_size]
        for i in range(0, len(texto), chunk_size)
    ]

    resultado = ""
    for parte in partes:
        resultado += translator.translate(parte) + " "

    return resultado

texto_es = traducir_texto_largo(texto_total)

#with open("transcripcion_es.txt", "w", encoding="utf-8") as f:
with open(properties["ARCHIVO_TRANSCRITO"], "w", encoding="utf-8") as f:

    f.write(texto_es)

print("✅ Traducción completada")
print("✅ Proceso terminado")
