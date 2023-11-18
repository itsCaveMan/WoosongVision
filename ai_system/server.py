'''
    This file is to test the API capability of the woosong vision AI detection system (apon a new trash bag loaded into the truck)
    This is a simple HTTP server that listens for POST requests on port 5000.
'''


import http.server
import json


class MyHandler(http.server.BaseHTTPRequestHandler):
    def do_POST(self):
        # Load the JSON payload from the request body
        payload_string = self.rfile.read(int(self.headers['Content-Length']))
        try:
            payload = json.loads(payload_string.decode('utf-8'))
        except json.JSONDecodeError:
            self.send_error(400, 'Invalid JSON payload')
            return

        print(payload)

        # Process the form data and send a response
        self.send_response(200)
        self.send_header('Content-type', 'text/html')
        self.end_headers()

httpd = http.server.HTTPServer(('localhost', 5000), MyHandler)
print('Starting server on port 5000...')
httpd.serve_forever()
