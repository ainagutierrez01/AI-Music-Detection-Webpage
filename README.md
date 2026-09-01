# Detector de música IA (TikTok) — demo web

Demo web del TFG "Benchmarking AI-Generated Music Detection Under TikTok
Audio Processing Conditions". Deixa pujar un fitxer d'àudio i triar entre els
4 checkpoints reals entrenats a la tesi (mateixa arquitectura CNN, dades
d'entrenament diferents):

- `700_model_tiktok.pth` / `600_model_tiktok.pth` — dataset propi de TikTok
- `model_sonics.pth` — enfocament SONICS
- `model_laura_cros_vila.pth` — enfocament Lara Cros-Vila (CLAP)

Els pesos NO estan inclosos en aquest zip (per pes): l'aplicació els
descarrega automàticament des de les Releases del repositori
(`ainagutierrez/AI-tiktok-detection`) la primera vegada que es fan servir, i
els guarda a `weights/` per a les properes peticions.

## Provar-ho en local

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
python app.py
```

Obre `http://localhost:5000`.

## Publicar-ho amb un enllaç públic (recomanat: Render, gratuït)

1. Puja aquesta carpeta a un repositori de GitHub (pot ser un de nou, no cal
   que sigui el del TFG).
2. Ves a [render.com](https://render.com) → *New +* → *Web Service* →
   connecta el repositori.
3. Render detectarà el `render.yaml` automàticament (build i start command
   ja configurats). Tria el pla **Free**.
4. Espera el primer desplegament (uns minuts, ja que instal·la PyTorch) i
   Render et donarà un enllaç del tipus `https://ai-music-tiktok-detector.onrender.com`.
5. Aquest és l'enllaç que pots posar al document del Pitching i ensenyar al
   clip audiovisual.

Nota: al pla gratuït de Render el servei "s'adorm" si no rep trànsit uns
minuts, i la primera petició després de dormir triga uns 30-60 segons a
respondre — normal en un pla gratuït, no és un error.

### Alternativa: Hugging Face Spaces

Si prefereixes Hugging Face Spaces (gratuït, sense "son"), cal fer un petit
canvi: crear un `Dockerfile` senzill que instal·li `requirements.txt` i
executi `app.py`, ja que Spaces amb SDK "Docker" simplement necessita
exposar el port 7860. Digues-m'ho si vols que te'l munti.

## Afegir el quart model (SpecTTTra / SONICS pretrained)

El detector original SONICS/SpecTTTra es descarrega des de Hugging Face Hub
(`awsaf49/sonics-spectttra-*`), no des de GitHub, i necessita el paquet
`sonics` (`pip install git+https://github.com/awsaf49/sonics.git`). No s'ha
inclòs en aquesta demo perquè l'entorn on s'ha construït no tenia accés a
Hugging Face, però un cop desplegat a Render o HF Spaces (que sí hi tenen
accés), es pot afegir reutilitzant `pretrained_detector.py` del repositori
original del TFG i afegint una cinquena entrada al diccionari `MODELS` de
`app.py`.
