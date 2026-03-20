# Paquet principal de l'application FastAPI
"""FastAPI application package."""

# Importe l'instance de l'application FastAPI
from api.app import app

# Expose uniquement l'objet app lors d'un import *
__all__ = ["app"]
