"""Hugging Face *Gradio-SDK* entrypoint (free CPU-basic tier, no Docker paywall).

The whole FastAPI backend is served as-is; a tiny Gradio landing page is mounted
at /ui only so the Space's "App" tab shows something. HF Spaces runs
``python app.py``, so the ``__main__`` block below binds uvicorn on port 7860.

Writable-dir note: on Spaces only /tmp is writable. HF_HOME and
MONITOR_STATE_PATH are pointed there via the Space's environment variables.
"""
import gradio as gr
import uvicorn

from app.backend.main import app as fastapi_app

with gr.Blocks(title="ViHSD Moderation API") as landing:
    gr.Markdown(
        "# 🛡️ ViHSD Moderation API\n"
        "FastAPI backend is live. Interactive API docs: **[/docs](/docs)** · "
        "health: **[/health](/health)**.\n\n"
        "This page exists only to satisfy the Gradio Space runtime; the real "
        "client is the Vercel frontend."
    )

# Mount the Gradio landing page onto the FastAPI app (not the reverse), so all
# FastAPI routes (/health, /predict, /showdown, ...) stay at the root.
app = gr.mount_gradio_app(fastapi_app, landing, path="/ui")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=7860)
