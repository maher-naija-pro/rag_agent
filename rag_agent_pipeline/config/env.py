"""Load environment variables from .env — must be imported first."""

# Module pour manipuler les chemins de fichiers
from pathlib import Path

# Fonction pour charger les variables d'environnement depuis un fichier .env
from dotenv import load_dotenv

# Résout le chemin du fichier .env relatif à la racine du projet (un niveau au-dessus de config/)
# Construit le chemin absolu vers le fichier .env
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
# Charge les variables d'environnement depuis le fichier .env
load_dotenv(dotenv_path=_ENV_PATH)
