import moviepy as mp
from faster_whisper import WhisperModel
import json

# ===================================
# CONFIGURACION
# ===================================
VIDEO_FILE = "uno.mp4"
AUDIO_FILE = "audio1.wav"

MODEL_SIZE = "medium"
# opciones:
# tiny | base | small | medium | large-v3

DEVICE = "cpu"  # "cuda" si tienes GPU NVIDIA

# ===================================
# 1️⃣ EXTRAER AUDIO
# ===================================
print("Extrayendo audio...")

video = mp.VideoFileClip(VIDEO_FILE)
video.audio.write_audiofile(
    AUDIO_FILE,
    fps=16000,
    codec="pcm_s16le"
)

# ===================================
# 2️⃣ CARGAR MODELO WHISPER
# ===================================
print("Cargando modelo Whisper...")

model = WhisperModel(
    MODEL_SIZE,
    device=DEVICE,
    compute_type="int8"  # optimizado CPU
)

# ===================================
# 3️⃣ TRANSCRIPCION + TRADUCCION
# ===================================
print("Transcribiendo...")

segments, info = model.transcribe(
    AUDIO_FILE,
    beam_size=5,
    task="translate"  # 👈 TRADUCE A INGLES AUTOMATICO
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

    # ----- SRT -----
    start = segment.start
    end = segment.end

    def format_time(t):
        h = int(t // 3600)
        m = int((t % 3600) // 60)
        s = int(t % 60)
        ms = int((t - int(t)) * 1000)
        return f"{h:02}:{m:02}:{s:02},{ms:03}"

    srt_content += f"{i}\n"
    srt_content += f"{format_time(start)} --> {format_time(end)}\n"
    srt_content += f"{segment.text.strip()}\n\n"

    # ----- JSON -----
    json_segments.append({
        "start": start,
        "end": end,
        "text": segment.text.strip()
    })

# ===================================
# 5️⃣ GUARDAR ARCHIVOS
# ===================================
print("Guardando resultados...")

with open("transcripcion.txt", "w", encoding="utf-8") as f:
    f.write(texto_total)

with open("subtitulos.srt", "w", encoding="utf-8") as f:
    f.write(srt_content)

with open("transcripcion.json", "w", encoding="utf-8") as f:
    json.dump(json_segments, f, indent=2, ensure_ascii=False)

print("✅ Proceso terminado")
