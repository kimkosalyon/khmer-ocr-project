import os
import re
import http.server
import socketserver
import sys

PORT = 8081
UPLOAD_DIR = "kmwiki_data"

class UploadHandler(http.server.BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        html = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Upload DuckDB File</title>
            <style>
                body { font-family: sans-serif; max-width: 500px; margin: 50px auto; padding: 20px; background: #f0f2f5; }
                .card { background: white; padding: 30px; border-radius: 8px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); }
                h2 { color: #1a73e8; margin-top: 0; }
                input[type=file] { display: block; margin: 20px 0; width: 100%; padding: 10px; border: 1px dashed #ccc; border-radius: 4px; }
                button { background: #1a73e8; color: white; border: none; padding: 10px 20px; border-radius: 4px; cursor: pointer; font-size: 16px; width: 100%; }
                button:hover { background: #1557b0; }
                .msg { margin-top: 20px; color: green; font-weight: bold; }
            </style>
        </head>
        <body>
            <div class="card">
                <h2>Upload .duckdb File</h2>
                <p>Select your DuckDB file to upload it directly into the <code>kmwiki_data/</code> folder.</p>
                <form method="POST" enctype="multipart/form-data" action="/upload">
                    <input type="file" name="file" accept=".duckdb" required />
                    <button type="submit">Upload File</button>
                </form>
            </div>
        </body>
        </html>
        """
        self.wfile.write(html.encode("utf-8"))

    def do_POST(self):
        if self.path == "/upload":
            content_type = self.headers.get('Content-Type')
            if not content_type or not content_type.startswith('multipart/form-data'):
                self.send_error(400, "Bad Request: Content-Type must be multipart/form-data")
                return

            try:
                boundary = content_type.split("boundary=")[1].encode()
            except IndexError:
                self.send_error(400, "Bad Request: Missing boundary in Content-Type")
                return
                
            content_length = int(self.headers.get('Content-Length', 0))
            
            # Read raw body
            body = self.rfile.read(content_length)
            
            # Split parts by boundary
            parts = body.split(b'--' + boundary)
            
            uploaded_file_data = None
            filename = "uploaded_database.duckdb"
            
            for part in parts:
                if b'Content-Disposition' in part and b'filename="' in part:
                    # Parse filename
                    match = re.search(br'filename="([^"]+)"', part)
                    if match:
                        filename = match.group(1).decode('utf-8')
                    try:
                        # Split headers from body in this part
                        header_end = part.index(b'\r\n\r\n')
                        uploaded_file_data = part[header_end + 4:]
                        # Strip trailing \r\n
                        if uploaded_file_data.endswith(b'\r\n'):
                            uploaded_file_data = uploaded_file_data[:-2]
                    except ValueError:
                        pass
                    break
            
            if uploaded_file_data:
                os.makedirs(UPLOAD_DIR, exist_ok=True)
                dest_path = os.path.join(UPLOAD_DIR, filename)
                with open(dest_path, "wb") as f:
                    f.write(uploaded_file_data)
                
                self.send_response(200)
                self.send_header("Content-Type", "text/html; charset=utf-8")
                self.end_headers()
                self.wfile.write(f"""
                <!DOCTYPE html>
                <html>
                <head>
                    <title>Upload Successful</title>
                    <style>
                        body {{ font-family: sans-serif; max-width: 500px; margin: 50px auto; padding: 20px; text-align: center; }}
                        .msg {{ color: green; font-size: 20px; font-weight: bold; margin-bottom: 20px; }}
                        a {{ color: #1a73e8; text-decoration: none; }}
                    </style>
                </head>
                <body>
                    <div class="msg">✓ File uploaded successfully!</div>
                    <p>Saved as: <code>{dest_path}</code></p>
                    <p>You can now close this tab and return to the chat interface.</p>
                </body>
                </html>
                """.encode("utf-8"))
            else:
                self.send_error(400, "Bad Request: No file data found in multipart body")

def main():
    # Allow port to be specified as argument
    port = PORT
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            pass

    Handler = UploadHandler
    # Allow address reuse
    socketserver.TCPServer.allow_reuse_address = True
    
    with socketserver.TCPServer(("", port), Handler) as httpd:
        print(f"============================================================")
        print(f"Starting Upload Server on port {port}")
        print(f"URL: http://localhost:{port}")
        print(f"============================================================")
        print("Press Ctrl+C to stop the server.")
        try:
            httpd.serve_forever()
        except KeyboardInterrupt:
            print("\nUpload server stopped.")

if __name__ == "__main__":
    main()
