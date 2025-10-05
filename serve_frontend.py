#!/usr/bin/env python3
"""
Simple HTTP server for frontend development.
Serves the frontend on port 3000.
"""

import http.server
import socketserver
import os
from pathlib import Path

PORT = 3000
DIRECTORY = "frontend"

class MyHTTPRequestHandler(http.server.SimpleHTTPRequestHandler):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, directory=DIRECTORY, **kwargs)

    def end_headers(self):
        # Add CORS headers
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        # Disable caching for development
        self.send_header('Cache-Control', 'no-store, no-cache, must-revalidate, max-age=0')
        self.send_header('Pragma', 'no-cache')
        self.send_header('Expires', '0')
        super().end_headers()

if __name__ == "__main__":
    os.chdir(Path(__file__).parent)

    print("="*60)
    print("Exoplanet System Detective - Frontend Server")
    print("="*60)
    print(f"\nServing frontend from: {DIRECTORY}/")
    print(f"Server running at: http://localhost:{PORT}")
    print("\nMake sure the backend API is running on port 8000!")
    print("  ./start_server.sh")
    print("\nOpen http://localhost:3000 in your browser")
    print("\nPress Ctrl+C to stop")
    print("="*60)
    print()

    with socketserver.TCPServer(("", PORT), MyHTTPRequestHandler) as httpd:
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\n\nShutting down server...")
            httpd.shutdown()
