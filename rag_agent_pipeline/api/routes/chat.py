# Endpoint de chat avec streaming token par token via SSE
"""POST /api/chat — true token-by-token LLM streaming (SSE)."""

# Active les annotations différées pour le typage
from __future__ import annotations

# Module de sérialisation/désérialisation JSON
import json
# File d'attente asynchrone pour le pont sync → async
import asyncio
# Manipulation de chemins de fichiers
from pathlib import Path

# Routeur FastAPI et gestion des erreurs HTTP
from fastapi import APIRouter, HTTPException
# Réponse HTTP en streaming
from fastapi.responses import StreamingResponse
# Types de messages LangChain
from langchain_core.messages import HumanMessage, SystemMessage

# Modèle de validation de la requête de chat
from api.schemas import ChatRequest
# État partagé : graphe, sessions et checkpointer
from api.state import graph, sessions, checkpointer
# Paramètres de configuration du pipeline
from config import (
    # Instance du modèle de langage
    LLM,
    # Nombre de documents à récupérer
    RETRIEVAL_K,
    # Coefficient de fusion pour la recherche hybride
    HYBRID_FUSION_ALPHA,
    # Seuil de similarité minimale
    SIMILARITY_THRESHOLD,
    # Fonction pour obtenir le store vectoriel Qdrant
    get_store,
)
# Fonctions de re-classement des résultats
from nodes.reranker import _build_reranker, _filter_by_score
# Seuil de score pour le re-classement
from config import RERANK_SCORE_THRESHOLD  # noqa: E402
# Fonction utilitaire pour créer un logger
from logger import get_logger
# Formatage des documents et template système
from nodes.generator import _format_docs, SYSTEM_TEMPLATE

# Crée un routeur pour les routes de chat
router = APIRouter()
# Crée un logger nommé pour ce module
log = get_logger("api.routes.chat")

# En-têtes HTTP pour le streaming SSE
_SSE_HEADERS = {
    # Désactive le cache navigateur
    "Cache-Control": "no-cache",
    # Maintient la connexion ouverte
    "Connection": "keep-alive",
    # Désactive le buffering côté proxy (nginx)
    "X-Accel-Buffering": "no",
}


# Déclare une route POST sur /api/chat
@router.post("/api/chat")
# Reçoit et valide la requête de chat
async def chat(req: ChatRequest):
    """Stream answer tokens in real time via SSE.

    Pipeline stages executed sequentially:
      1. Query rewriting (improve retrieval semantics)
      2. Retrieval from Qdrant (hybrid dense + sparse search)
      3. Reranking candidates
      4. LLM streaming — tokens yielded as they are generated
      5. Graph invocation to persist conversation in checkpointer
    """
    # Récupère la session correspondante
    session = sessions.get(req.session_id)
    # Vérifie que la session existe
    if not session:
        # Erreur 404 si session introuvable
        raise HTTPException(404, "Session not found. Ingest a PDF first.")
    # Vérifie que la question n'est pas vide
    if not req.question.strip():
        # Erreur 400 si question vide
        raise HTTPException(400, "Question cannot be empty")

    # Récupère l'identifiant de fil de conversation
    thread_id = session["thread_id"]
    # Extrait le nom du fichier source pour filtrer dans Qdrant
    qdrant_source = (
        Path(session.get("file_path", "")).name
        if session.get("file_path")
        else ""
    )
    # Configuration du checkpointer avec le thread_id
    config = {"configurable": {"thread_id": thread_id}}

    # Journalise la requête de chat
    log.info("Chat request (session=%s): '%s'", req.session_id, req.question[:80])

    # ── Sentinel indiquant la fin du flux dans la queue ──
    _DONE = object()

    # File d'attente asynchrone reliant le thread sync au générateur async.
    # Le pipeline RAG (rewrite, retrieval, rerank, LLM) est entièrement
    # synchrone. Si on l'exécute dans un async generator, il bloque la
    # boucle événementielle d'Uvicorn et empêche l'envoi des octets SSE.
    # La solution : exécuter le pipeline dans un thread via asyncio.to_thread()
    # et pousser chaque événement SSE dans une asyncio.Queue que le
    # générateur async consomme sans bloquer.
    queue: asyncio.Queue[str | object] = asyncio.Queue()

    def _run_pipeline():
        """Exécute le pipeline RAG complet dans un thread séparé.

        Pousse chaque événement SSE (token, sources, done, error)
        dans la queue partagée, terminé par le sentinel _DONE.
        """
        try:
            # ── Stage 1: Rewrite query for better retrieval ──────────
            # Import paresseux du module de réécriture
            from nodes.query_rewriter import rewrite_query

            # Prépare l'état pour la réécriture de la requête
            rewrite_state: dict = {
                # Question originale de l'utilisateur
                "question": req.question,
                # Champ pour la question originale (avant réécriture)
                "original_question": "",
                # Historique des messages de conversation
                "messages": [],
                # Source du document pour le filtrage
                "source": qdrant_source,
            }
            # Load conversation history from the checkpoint
            # Récupère le checkpoint existant
            checkpoint = checkpointer.get(config)
            # Vérifie si un historique existe
            if checkpoint and "channel_values" in checkpoint:
                # Charge l'historique des messages
                rewrite_state["messages"] = checkpoint["channel_values"].get(
                    "messages", []
                )

            # Réécrit la question pour améliorer la recherche
            rewrite_result = rewrite_query(rewrite_state)
            # Utilise la question réécrite ou l'originale
            search_question = rewrite_result.get("question", req.question)

            # ── Stage 2: Retrieve candidates from Qdrant ─────────────
            # Paramètres de recherche : nombre de résultats
            search_kwargs: dict = {"k": RETRIEVAL_K}
            # Applique le seuil de similarité si défini
            if SIMILARITY_THRESHOLD > 0.0:
                search_kwargs["score_threshold"] = SIMILARITY_THRESHOLD
            # Ajuste le coefficient de fusion hybride si différent de la valeur par défaut
            if HYBRID_FUSION_ALPHA != 0.5:
                search_kwargs["alpha"] = HYBRID_FUSION_ALPHA

            # Filter to current document only
            # Filtre les résultats pour ne garder que le document courant
            if qdrant_source:
                # Importe les modèles de filtrage Qdrant
                from qdrant_client.models import FieldCondition, MatchValue, Filter
                # Crée un filtre sur le champ source des métadonnées
                search_kwargs["filter"] = Filter(must=[
                    FieldCondition(key="metadata.source", match=MatchValue(value=qdrant_source))
                ])

            # Crée un retriever à partir du store vectoriel
            base_retriever = get_store().as_retriever(
                search_type="similarity", search_kwargs=search_kwargs,
            )
            # Exécute la recherche et récupère les documents candidats
            candidates = base_retriever.invoke(search_question)

            # ── Stage 3: Rerank candidates ───────────────────────────
            # Instancie le modèle de re-classement
            reranker = _build_reranker()
            # Re-classe les documents candidats par pertinence
            context_docs = list(
                reranker.compress_documents(candidates, search_question)
            )
            # Filtre les documents sous le seuil de score
            context_docs = _filter_by_score(context_docs, RERANK_SCORE_THRESHOLD)

            # Extrait et trie les numéros de pages sources uniques
            source_pages = sorted(
                set(
                    d.metadata.get("page", 0)
                    for d in context_docs
                    if d.metadata.get("page")
                )
            )

            # ── Stage 4: Stream LLM answer token by token ───────────
            # Formate les documents en texte contextuel pour le LLM
            context_str = _format_docs(context_docs)
            # Récupère le checkpoint pour l'historique
            checkpoint = checkpointer.get(config)
            # Initialise la liste de l'historique de conversation
            history: list = []
            # Vérifie si un historique existe
            if checkpoint and "channel_values" in checkpoint:
                # Récupère les messages précédents
                msgs = checkpoint["channel_values"].get("messages", [])
                # Exclut le dernier message pour éviter les doublons
                history = msgs[:-1] if msgs else []

            # Construit le prompt complet pour le LLM
            prompt = (
                # Message système avec le contexte
                [SystemMessage(content=SYSTEM_TEMPLATE.format(context=context_str))]
                # Ajoute l'historique de conversation
                + history
                # Ajoute la question de l'utilisateur
                + [HumanMessage(content=req.question)]
            )

            # Accumule la réponse complète au fil du streaming
            full_answer = ""
            # Itère sur chaque token généré par le LLM
            for chunk_token in LLM.stream(prompt):
                # Extrait le texte du token
                tok = chunk_token.content
                # Ignore les tokens vides
                if tok:
                    # Ajoute le token à la réponse complète
                    full_answer += tok
                    # Pousse le token dans la queue SSE
                    queue.put_nowait(
                        f"data: {json.dumps({'type': 'token', 'content': tok})}\n\n"
                    )

            # ── Stage 5: Persist conversation in checkpointer ────────
            # Importe le type de message IA et l'utilitaire de checkpoint
            from langchain_core.messages import AIMessage
            from langgraph.checkpoint.base import empty_checkpoint

            try:
                # Prépare la configuration de sauvegarde avec checkpoint_ns
                # dans "configurable" (là où InMemorySaver l'attend)
                save_config = {
                    "configurable": {
                        "thread_id": thread_id,
                        "checkpoint_ns": "",
                    }
                }
                # Récupère le checkpoint actuel
                checkpoint = checkpointer.get(save_config)
                # Initialise la liste des messages précédents
                prev_msgs = []
                # Vérifie l'existence de messages précédents
                if checkpoint and "channel_values" in checkpoint:
                    # Récupère les messages sauvegardés
                    prev_msgs = checkpoint["channel_values"].get("messages", [])
                # Ajoute la question et la réponse à l'historique
                new_msgs = prev_msgs + [
                    # Message de l'utilisateur
                    HumanMessage(content=req.question),
                    # Réponse générée par le LLM
                    AIMessage(content=full_answer),
                ]
                # Construit un checkpoint valide avec la structure attendue
                # par InMemorySaver.put() (id, ts, v, channel_versions, etc.)
                new_cp = empty_checkpoint()
                new_cp["channel_values"] = {"messages": new_msgs}
                new_cp["channel_versions"] = {"messages": len(new_msgs)}
                # Sauvegarde le nouvel état dans le checkpointer
                checkpointer.put(
                    save_config,
                    # Checkpoint complet avec tous les champs requis
                    new_cp,
                    # Métadonnées du checkpoint
                    {"source": "input", "step": len(new_msgs), "writes": {}},
                    # Nouvelles versions pour le suivi des canaux
                    {"messages": len(new_msgs)},
                )
            # Capture les erreurs de sauvegarde
            except Exception as save_err:
                # Journalise l'échec sans interrompre le flux
                log.warning("Failed to save conversation: %s", save_err)

            # Emit source pages and final done event
            # Envoie les pages sources si des références ont été trouvées
            if source_pages:
                # Événement SSE avec les numéros de pages
                queue.put_nowait(
                    f"data: {json.dumps({'type': 'sources', 'pages': source_pages})}\n\n"
                )

            # Événement SSE final avec la réponse complète
            queue.put_nowait(
                f"data: {json.dumps({'type': 'done', 'answer': full_answer})}\n\n"
            )

        # Capture toute erreur survenue pendant le pipeline
        except Exception as e:
            # Journalise l'erreur
            log.error("Chat error (session=%s): %s", req.session_id, e)
            # Envoie un événement SSE d'erreur
            queue.put_nowait(
                f"data: {json.dumps({'type': 'error', 'content': str(e)})}\n\n"
            )
        finally:
            # Signale la fin du flux au générateur async
            queue.put_nowait(_DONE)

    async def event_stream():
        """Générateur async qui consomme la queue sans bloquer la boucle.

        Le pipeline sync tourne dans un thread séparé (asyncio.to_thread)
        et pousse les événements SSE dans la queue. Ce générateur les
        récupère avec await queue.get() — coopératif, donc Uvicorn peut
        envoyer chaque chunk dès qu'il arrive.
        """
        # Lance le pipeline sync dans un thread du pool par défaut
        task = asyncio.get_event_loop().run_in_executor(None, _run_pipeline)
        # Consomme les événements SSE de la queue jusqu'au sentinel
        while True:
            item = await queue.get()
            if item is _DONE:
                break
            yield item
        # Attend la fin du thread pour propager les exceptions éventuelles
        await task

    # Retourne une réponse HTTP en streaming
    # Configure le type MIME SSE et les en-têtes
    return StreamingResponse(
        event_stream(), media_type="text/event-stream", headers=_SSE_HEADERS,
    )
