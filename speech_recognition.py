import vosk
import sounddevice as sd
import numpy as np
import json
import os
import subprocess
import webbrowser
import shutil
from pathlib import Path
import time

# Ramdisk-Pfad und Cache-Verwaltung
RAMDISK_PATH = "R:/vosk_cache"
MODEL_CACHE = os.path.join(RAMDISK_PATH, "vosk-model-de-0.21")
RNNLM_DISABLED_FLAG = os.path.join(RAMDISK_PATH, ".rnnlm_disabled")

def prepare_model_cache():
    downloads_path = str(Path.home() / "Downloads")
    source_model = os.path.join(downloads_path, "vosk-model-de-0.21")
    
    # Erstelle Cache-Verzeichnis falls nicht vorhanden
    if not os.path.exists(RAMDISK_PATH):
        os.makedirs(RAMDISK_PATH)
    
    # Kopiere Modell in Ramdisk wenn noch nicht vorhanden
    if not os.path.exists(MODEL_CACHE):
        print("\nKopiere Modell in Ramdisk-Cache...")
        shutil.copytree(source_model, MODEL_CACHE)
        print("Modell erfolgreich in Cache kopiert!")
    
    # Deaktiviere RNNLM für schnelleres Laden (nur einmal)
    if not os.path.exists(RNNLM_DISABLED_FLAG):
        rnnlm_path = os.path.join(MODEL_CACHE, "rnnlm", "final.raw")
        if os.path.exists(rnnlm_path):
            rnnlm_backup = rnnlm_path + ".backup"
            if not os.path.exists(rnnlm_backup):
                print("\nDeaktiviere RNNLM für schnelleres Laden...")
                shutil.move(rnnlm_path, rnnlm_backup)
                # Erstelle Flag-Datei
                with open(RNNLM_DISABLED_FLAG, 'w') as f:
                    f.write('RNNLM disabled for faster loading')
    
    print("\nLade Modell...")
    return vosk.Model(MODEL_CACHE)

# Modell aus Cache laden
try:
    model = prepare_model_cache()
    print("Modell erfolgreich geladen!")
except Exception as e:
    print(f"Fehler beim Laden des Modells: {e}")
    exit(1)

# Verfügbare Audiogeräte anzeigen
print("\nVerfügbare Audiogeräte:")
print(sd.query_devices())
default_device = sd.query_devices(kind='input')
print(f"\nStandard-Eingabegerät: {default_device['name']}")

# Audioeinstellungen optimieren
samplerate = 16000  # Feste Samplerate für bessere Performance
channels = 1
device = default_device['index']
blocksize = 2000  # Noch kleinerer Blocksize für schnellere Reaktion

print(f"\nAudio-Konfiguration:")
print(f"Samplerate: {samplerate}")
print(f"Gerät-Index: {device}")
print(f"Blocksize: {blocksize}")

# RecognizeMicrophone erstellen
print("\nInitialisiere Spracherkennung...")
rec = vosk.KaldiRecognizer(model, samplerate)
print("Spracherkennung bereit!")

def execute_command(text):
    text = text.lower()
    print(f"\nVerarbeite Befehl: {text}")
    
    # Befehle definieren
    if "browser" in text or "internet" in text:
        webbrowser.open("https://www.duckduckgo.com")
        print("Öffne Browser...")
        
    elif "notepad" in text or "editor" in text:
        subprocess.Popen("notepad.exe")
        print("Öffne Notepad...")
        
    elif "rechner" in text or "taschenrechner" in text:
        subprocess.Popen("calc.exe")
        print("Öffne Rechner...")
        
    elif "beenden" in text or "stop" in text:
        print("Programm wird beendet...")
        return False
        
    return True

def is_silent(audio_data, threshold=300):  # Niedrigerer Schwellwert
    """Prüft ob der Audio-Input still ist"""
    # Konvertiere bytes zu numpy array
    audio_array = np.frombuffer(audio_data, dtype=np.int16)
    # Berechne RMS (Root Mean Square) als Maß für die Lautstärke
    rms = np.sqrt(np.mean(np.square(audio_array)))
    return rms < threshold

def callback(indata, frames, time_info, status):
    global running, last_active_time
    if status:
        print(f"Status: {status}")
    
    try:
        # Konvertiere numpy array zu bytes
        audio_data = bytes(indata)
        
        # Debug-Output für Stille-Erkennung
        current_time = time.time()
        if is_silent(audio_data):
            if current_time - last_active_time > 0.3:  # Kürzere Wartezeit (300ms)
                print(".", end="", flush=True)
                last_active_time = current_time
        else:
            print("\nAudio-Input erkannt!")
            last_active_time = current_time
        
        if rec.AcceptWaveform(audio_data):
            result = json.loads(rec.Result())
            if result["text"]:
                print("\nErkannt:", result["text"])
                running = execute_command(result["text"])
    except Exception as e:
        print(f"Fehler in callback: {e}")

# Stream starten
running = True
last_active_time = time.time()  # Für Stille-Erkennung
print("\nVerfügbare Befehle: browser, notepad, rechner, beenden")
print("Sprechen Sie jetzt... (Strg+C zum Beenden)")
print("Debug: '.' = Stille erkannt")

try:
    with sd.RawInputStream(samplerate=samplerate, 
                         blocksize=blocksize,
                         device=device,
                         dtype=np.int16, 
                         channels=channels, 
                         callback=callback):
        print("\nAudio-Stream gestartet!")
        while running:
            sd.sleep(20)  # Noch kürzere Sleep-Zeit
            
except Exception as e:
    print(f"\nFehler beim Starten des Audio-Streams: {e}")
finally:
    print("\nProgramm beendet.") 