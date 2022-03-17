# Présentation

Ce projet fait partie du challenge de dernière année du master SISE à Lyon 2. Son but était de créer un outil de reconnaissance facial en deux jours.

# Installation

Pour installer notre programme :

●	Ouvrir un terminal anaconda prompt,

●	se placer dans le dossier où se trouve le fichier téléchargé environment.yml,

●	lancer la commande : conda env create -f environment.yml,

●	activer votre environnement : conda activate gest

Votre nouvel environnement est créé et activé.

●	mettez-vous dans le répertoire du fichier dézippé puis :

●	lancer l’application avec la commande : python natacha.py

●	Pour lancer l’application de commande gestuelle, il suffit de dire ou d’écrire “natacha contrôle” (veuillez attendre 2 secondes après sa phrase de bienvenue).

●	Placez votre main dans l’écran en forme de “tchuss”, votre main et vos doigts devraient être détectés et 3 points roses doivent se former entre votre index et votre majeur. Pour simuler un clic afin d’ouvrir la reconnaissance faciale, il suffit de rapprocher les 2 doigts.

La fenêtre devrait “freezer” en attendant l’ouverture de l’application de reconnaissance faciale. Vous devriez voir le chargement des processus dans la fenêtre d’invite de commande anaconda.

Une fois la reconnaissance faciale ouverte, la fenêtre de l’application de reconnaissance gestuelle devrait pouvoir se fermer facilement en cliquant sur la croix rouge.

Pour arrêter la totalité des processus, il faut revenir sur le chatbot et dire ou écrire “merci natacha”.
Vous pouvez ensuite fermer toutes les fenêtres.

Vous pourrez aussi retrouver la vidéo enregistrée au même endroit que celui des fichiers dézippés.


Il est fort possible que vous rencontriez des bugs (notamment sur la fermeture des diverses applications), le code n’est clairement pas optimal.
