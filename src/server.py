from http.server import HTTPServer, BaseHTTPRequestHandler
import json
from os import listdir
from predictHashtags import predict
from trainNN import RNN3


class Handler(BaseHTTPRequestHandler):

    def list_get(self):
        self.wfile.write(json.dumps(listdir('models')).encode('utf8'))

    def predict_hashtags_post(self):
        data = self._parse_post()
        try:
            texts = data['texts']
            preds = predict('../models/mod3.mod', '../doc/vocab.json', '../doc/vocabH.json', texts, '../doc/wordoc')
            self.wfile.write(json.dumps(preds).encode('utf8'))
        except KeyError:
            self.send_error(25, 'Wrong json format')

    def _set_headers(self):
        self.send_response(200)
        self.send_header('Content-type', 'application/json')
        self.end_headers()

    def _redirect(self, type):
        try:
            method = getattr(Handler, self.path[1:] + '_' + type)
            self._set_headers()
            method(self)
        except AttributeError:
            self.send_error(404)

    def do_HEAD(self):
        self._set_headers()

    def _parse_post(self):
        data_string = self.rfile.read(int(self.headers['Content-Length'])).decode('utf-8')
        return json.loads(data_string)

    def do_POST(self):
        self._redirect('post')

    def do_GET(self):
        self._redirect('get')


def run(server_c=HTTPServer, handler=Handler, port=8080):
    server_address = ('', port)
    httpd = server_c(server_address, handler)
    print ('Starting web api...')
    httpd.serve_forever()


run()
