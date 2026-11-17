"""Read-only local viewer; it never starts training or modifies a Q table."""
from http.server import SimpleHTTPRequestHandler, ThreadingHTTPServer
from functools import partial


def serve(root, port=8765):
    class Handler(SimpleHTTPRequestHandler):
        def do_GET(self):
            if self.path == "/":
                self.send_response(302)
                self.send_header("Location", "/gui/")
                self.end_headers()
                return
            return super().do_GET()
    handler = partial(Handler, directory=str(root))
    with ThreadingHTTPServer(("127.0.0.1", port), handler) as server:
        print(f"FrozenLake viewer: http://127.0.0.1:{port}/gui/", flush=True)
        server.serve_forever()
