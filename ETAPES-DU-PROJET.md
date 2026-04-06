# Création du dossier Certification
# Création du repo github Certification-Thomas DINH et push d'un premier README pour avoir quelque chose :

echo "# Certification-Thomas-DINH" >> README.md    
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:DINH-Thomas/Certification-Thomas-DINH.git
git push -u origin main

# Création d'une branche « Thomas » pour travailler dessus et pusher ensuite sur le main ensuite.
# Changement du répertoire de travail dans « Thomas »

git checkout Thomas

# Création d’un nouvel environnement virtuel et activation de ce dernier.
pyenv virtualenv 3.11.8 Certification                                                  
pyenv local Certification  

# Installation de poetry dans le nouvel environnement
pip install poetry
poetry init

# Création du .gitignore


