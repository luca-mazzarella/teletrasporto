# Teletrasporto (Shunkan Idō) 🫰

Effetto "teletrasporto istantaneo" di Goku sulla webcam: porta **indice e medio uniti alla fronte**
e sparisci con sfarfallio, glitch e scia di luce. Rifai il gesto per **ricomparire**.

## Installazione
```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```
I modelli MediaPipe (mano, volto, segmentazione) vengono scaricati da soli in `models/` al primo avvio.

## Uso
```bash
python teletrasporto.py
```
1. **Per un risultato migliore:** esci dall'inquadratura e premi `b` per catturare lo sfondo pulito
   (altrimenti il programma lo ricostruisce man mano mentre ti muovi).
2. Rientra, porta indice + medio alla fronte (anulare e mignolo piegati) e tienili lì un attimo.
3. Rifai il gesto per riapparire.

| Tasto | Azione |
|---|---|
| `b` | cattura lo sfondo |
| `spazio` | attiva l'effetto a mano |
| `d` | debug: punti della mano, zona fronte, barra di carica |
| `h` | mostra/nasconde l'help |
| `q` / `ESC` | esci |

Opzioni utili:
- `--camera 1` usa un'altra webcam (oppure un file video)
- `--proc-width 640` più fluido su PC lenti
- `--output video.mp4` registra il risultato
- `--virtual-cam` invia l'output a una webcam virtuale (Zoom, Meet, OBS, Discord) — richiede `pip install pyvirtualcam`
  e OBS installato (Windows/macOS) oppure `v4l2loopback` (Linux)

## Come funziona
- **MediaPipe Hand Landmarker** → controlla che indice e medio siano tesi e uniti, anulare e mignolo piegati.
- **MediaPipe Face Detector** → stima la posizione della fronte (sopra gli occhi) e verifica che le dita ci siano sopra.
- **Selfie Segmenter** → ritaglia la tua figura; lo sfondo viene memorizzato per "vedere attraverso" di te.
- **Effetto** (`render()`): la figura sfarfalla, si riempie di luce, subisce glitch orizzontali e aberrazione
  cromatica, si comprime in una sottile striscia verticale con linee di velocità e poi svanisce.
  La ricomparsa è la stessa animazione al contrario.

Le durate e le soglie si regolano in cima al file (`VANISH_TIME`, `APPEAR_TIME`, `HOLD_TIME`, `COOLDOWN`).
