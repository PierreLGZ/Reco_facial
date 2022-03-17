import pyttsx3
import speech_recognition as sr
from datetime import date
import time
import webbrowser
import datetime
from pynput.keyboard import Key, Controller
import sys
import os
from os import listdir
from os.path import isfile, join
import smtplib
import app
from threading import Thread
import controle
import cv2

# Intialisation des objets
r = sr.Recognizer()
keyboard = Controller()
engine = pyttsx3.init('sapi5')
engine = pyttsx3.init()
voices = engine.getProperty('voices')
engine.setProperty('voice', voices[0].id)

# Instanciation des variables
file_exp_status = False
files =[]
path = ''
is_awake = True  # Statut du bot

# Fonctions
def reply(audio):
    app.ChatBot.addAppMsg(audio)
    engine.say(audio)
    engine.runAndWait()

def wish():
    reply("Bonjour, je m'appelle Robot, comment puis-je vous aider ?")

# Paramètres du micro
with sr.Microphone() as source:
        r.energy_threshold = 100
        r.dynamic_energy_threshold = False

# Cast de l'audio en string
def record_audio():
    with sr.Microphone() as source:
        r.pause_threshold = 0.8
        voice_data = ''
        audio = r.listen(source, phrase_time_limit=5)

        try:
            voice_data = r.recognize_google(audio, language='fr-FR')
        except sr.RequestError:
            reply("Désolé, j'ai un problème de réseau. Pouvez-vous vérifier votre connexion ?")
        except sr.UnknownValueError:
            print("Je ne comprends pas !")
            pass
        return voice_data.lower()

# Exécution des commandes
def respond(voice_data):
    global file_exp_status, files, is_awake, path
    voice_data.replace('robot','')
    app.eel.addUserMsg(voice_data)

    if is_awake==False:
        if 'réveille-toi' in voice_data:
            is_awake = True
            wish()

    # Contrôles simples
    elif 'bonjour' in voice_data:
        wish()

    elif 'dédicace' in voice_data:
        reply("Ricco est notre héros !")

    elif ('bye' in voice_data) or ('by' in voice_data):
        reply("A plus tard mes B.G.!")
        is_awake = False

    elif ('merci' in voice_data):
        if controle.Controle.gc_mode:
            controle.Controle.gc_mode = 0
        app.ChatBot.close()
        cv2.waitKey(10)
        cv2.destroyAllWindows()
        sys.exit()
        
    # Contrôles dynamiques
    elif 'contrôle' in voice_data:
        if controle.Controle.gc_mode:
            reply("Vous avez déjà le contrôle")
        else:
            c = controle.Controle()
            t = Thread(target = c.start)
            t.start()
            reply("Lancement en cours !")

    elif 'stop' in voice_data:
        if controle.Controle.gc_mode:
            controle.Controle.gc_mode = 0
            reply("Fermeture de l'application")
        else:
            reply("L'application est déjà inactive")
                   
    else: 
        reply("Je ne suis pas programmée pour faire ça !")

# Exécution du chatbot
t1 = Thread(target = app.ChatBot.start)
t1.start()

while not app.ChatBot.started:
    time.sleep(0.5)

wish()
voice_data = None
while True:
    if app.ChatBot.isUserInput():
        # Récupération de la saisie clavier
        voice_data = app.ChatBot.popUserInput()
    else:
        # Récupération du vocal
        voice_data = record_audio()

    # Reconnaissance vocale
    if 'robot' in voice_data:
        try:
            respond(voice_data)
        except SystemExit:
            reply("A bientôt")
            break
        except:
            print("Une exception s'est produite pendant la fermeture.") 
            break
        


