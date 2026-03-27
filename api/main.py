from fastapi import FastAPI
from api.search_api import router as search_router
from api.index_api import router as index_router

app = FastAPI(title="Face Search API")

app.include_router(search_router)
app.include_router(index_router)