from fastapi import FastAPI
from apscheduler.schedulers.background import BackgroundScheduler
from contextlib import asynccontextmanager
import uvicorn
from src.api.main import lifespan

app=FastAPI(lifespan=lifespan)

if __name__ =="__main__":
    uvicorn.run("main:app")

