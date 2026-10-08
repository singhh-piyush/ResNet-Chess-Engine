"""Compatibility entry point for the backend."""
from backend.app import app

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=7860)
