import asyncio
import contextlib
import json
import logging
import os
import threading
from pathlib import Path
import chess
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from engine.runtime import Runtime
from engine.search import Search

runtime = None
load_error = None
busy = threading.Lock()


@contextlib.asynccontextmanager
async def lifespan(app):
    global runtime, load_error
    try:
        runtime = Runtime(os.getenv('MODEL_MANIFEST', 'models/release.json'))
        load_error = None
    except Exception as exc:
        runtime = None
        load_error = str(exc)
        logging.exception('Model initialization failed')
    yield


app = FastAPI(title='Personal Chess Engine', lifespan=lifespan)
origins = [x.strip() for x in os.getenv('CORS_ORIGINS','http://localhost:5173').split(',') if x.strip()]
app.add_middleware(CORSMiddleware, allow_origins=origins, allow_methods=['GET','POST'], allow_headers=['Content-Type'])


class FenRequest(BaseModel):
    fen: str = Field(max_length=150)
    starting_fen: str | None = Field(default=None,max_length=150)
    moves: list[str] = Field(default_factory=list,max_length=1000)


def position(request):
    try:
        board = chess.Board(request.fen)
        if not board.is_valid():
            raise ValueError('Invalid chess position')
        if request.moves or request.starting_fen:
            replay = chess.Board(request.starting_fen or chess.STARTING_FEN)
            if not replay.is_valid():
                raise ValueError('Invalid starting position')
            for uci in request.moves:
                replay.push_uci(uci)
            if replay.fen()!=board.fen():
                raise ValueError('Move history does not match FEN')
            board = replay
        return board
    except ValueError as exc:
        raise HTTPException(422,str(exc)) from exc


def require_runtime():
    if runtime is None:
        raise HTTPException(503,'Model unavailable; check readiness')
    return runtime


@app.get('/health')
def health():
    return {'status':'ok'}


@app.get('/ready')
def ready():
    if runtime is None:
        raise HTTPException(503,'Model unavailable')
    return {'status':'ready','model_version':runtime.version,'policy_size':runtime.policy_size}


@app.post('/predict')
async def predict(request: FenRequest):
    board = position(request)
    engine = require_runtime()
    if not busy.acquire(blocking=False):
        raise HTTPException(429,'Engine busy; retry after the current search')
    # Shield ensures cancellation never frees the engine while its thread still runs.
    cancelled = threading.Event()
    def cancellable_work():
        try:
            return Search(engine,cancelled=cancelled.is_set).run(board)
        finally:
            busy.release()
    try:
        return await asyncio.shield(asyncio.create_task(asyncio.to_thread(cancellable_work)))
    except asyncio.CancelledError:
        cancelled.set()
        raise


@app.post('/predict/stream')
async def stream(request: FenRequest):
    board = position(request)
    engine = require_runtime()
    if not busy.acquire(blocking=False):
        raise HTTPException(429,'Engine busy; retry after the current search')
    loop = asyncio.get_running_loop()
    queue = asyncio.Queue()
    cancelled = threading.Event()
    def send(event,data):
        loop.call_soon_threadsafe(queue.put_nowait,(event,data))
    def work():
        try:
            send('started',{'message':'Search started','model_version':engine.version})
            result = Search(engine,cancelled=cancelled.is_set).run(board,lambda p:send('progress',p))
            send('result',result)
        except Exception:
            logging.exception('Search failed')
            send('error',{'message':'Search interrupted or unavailable'})
        finally:
            busy.release()
    task = asyncio.create_task(asyncio.to_thread(work))
    async def events():
        try:
            while True:
                event,data = await queue.get()
                yield f'event: {event}\ndata: {json.dumps(data)}\n\n'
                if event in ('result','error'):
                    break
        finally:
            cancelled.set()
            # The worker owns the lock until it acknowledges cancellation.
            await asyncio.shield(task)
    return StreamingResponse(events(),media_type='text/event-stream',headers={'Cache-Control':'no-cache','X-Accel-Buffering':'no'})


class DrawRequest(FenRequest):
    user_side: str = Field(default='white',pattern='^(white|black)$')


@app.post('/offer_draw')
async def offer_draw(request: DrawRequest):
    board = position(request)
    engine = require_runtime()
    if not busy.acquire(blocking=False):
        raise HTTPException(429,'Engine busy')
    def work():
        try:
            _, value = engine.evaluate([board])[0]
            white_value = value if board.turn else -value
            bot_value = -white_value if request.user_side=='white' else white_value
            accepted = bot_value<=20
            return {'accepted':accepted,'message':'Draw agreed.' if accepted else "I would like to play on."}
        finally:
            busy.release()
    return await asyncio.shield(asyncio.create_task(asyncio.to_thread(work)))


build = Path(os.getenv('FRONTEND_BUILD','chess-frontend/dist'))
if (build/'index.html').is_file():
    for name in ('assets','TakenPiecesSVG'):
        if (build/name).is_dir():
            app.mount('/'+name,StaticFiles(directory=build/name),name=name)
    @app.get('/')
    async def index():
        return FileResponse(build/'index.html')
