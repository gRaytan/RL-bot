#!/usr/bin/env python3
"""Run the Harel Insurance Chatbot API server."""

import argparse
import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import uvicorn


def main():
    parser = argparse.ArgumentParser(description="Run Harel Chatbot API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")
    parser.add_argument("--workers", type=int, default=1, help="Number of workers")
    
    args = parser.parse_args()
    
    print(f"""
╔══════════════════════════════════════════════════════════════╗
║           Harel Insurance Chatbot API Server                 ║
╠══════════════════════════════════════════════════════════════╣
║  Endpoints:                                                  ║
║    POST /chat          - Send a message                      ║
║    GET  /health        - Health check                        ║
║    GET  /sessions      - List sessions                       ║
║    GET  /sessions/{{id}} - Get session details                 ║
║                                                              ║
║  Documentation:                                              ║
║    http://{args.host}:{args.port}/docs     - Swagger UI                ║
║    http://{args.host}:{args.port}/redoc    - ReDoc                     ║
╚══════════════════════════════════════════════════════════════╝
    """)
    
    uvicorn.run(
        "src.api.app:app",
        host=args.host,
        port=args.port,
        reload=args.reload,
        workers=args.workers if not args.reload else 1,
    )


if __name__ == "__main__":
    main()

