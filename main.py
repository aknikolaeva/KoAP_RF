from fastapi import FastAPI
import argparse
from core.retriever import QueryRequest, Retriever
import logging


app = FastAPI()

retriever = Retriever()


@app.on_event("startup")
async def startup_event():
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(filename)s:%(lineno)s - %(message)s",
        level=logging.INFO,
    )


@app.post("/ask")
async def ask_endpoint(request: QueryRequest):
    response = await retriever.ask(request)
    return response
