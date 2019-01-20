#!/usr/bin/python
from http.server import BaseHTTPRequestHandler, HTTPServer
# from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from os import curdir, sep, path
import cgi
import json

DOC_ROOT = path.join(path.dirname(path.realpath(__file__)), 'public')
PORT_NUMBER = 8080
DBG_LOG = False

from pose_estimation.infer_coco_simple import init_model, load_model, infer, release_model

model = None
def _prepare_model():
  global model
  if model:
    return
  _ = init_model()
  model = load_model()

def _release_model():
  global model
  if not model:
    return
  del model
  model = None
  release_model()


#This class will handles any incoming request from
#the browser 
class MyHandler(BaseHTTPRequestHandler):
  def inference_request(self, data_path):
      if DBG_LOG: print("data_path: %s" % data_path)
      _prepare_model()
      all_preds = infer(path.dirname(data_path), data_path, model)
      return {
        'status': 200,
        'message': "",
        'result': all_preds.tolist()
      }

	#Handler for the GET requests
  def do_GET(self):
    if self.path=="/":
      self.path="/index.html"

    #Check the file extension required and
    #set the right mime type
    msg = None
    if self.path.endswith(".json"):
      data_path = self.path
      if data_path.startswith('/C/'): data_path = data_path.replace('/C/', 'C:/', 1)
      elif data_path.startswith('/data/'): data_path = '.' + data_path
      msg = self.inference_request(data_path)
    elif self.path.endswith(".html"):
      mimetype='text/html'
    elif self.path.endswith(".jpg"):
      mimetype='image/jpg'
    elif self.path.endswith(".gif"):
      mimetype='image/gif'
    elif self.path.endswith(".js"):
      mimetype='application/javascript'
    elif self.path.endswith(".css"):
      mimetype='text/css'
    else:
      if self.path == '/closepose':
        _release_model()
        msg = {
          'status': 200,
          'message': 'released...'
        }
        self.send_header('Content-type', 'text')
      else:
        return # invalid request

    if msg:
      self.send_response(200)
      self.send_header('Content-type', 'application/json')
      self.end_headers()
      self.wfile.write(json.dumps(msg).encode())
    else:
      # serve file
      try:
        #Open the static file requested and send it
        f = open(DOC_ROOT + self.path, 'rb') 
        self.send_response(200)
        self.send_header('Content-type',mimetype)
        self.end_headers()
        self.wfile.write(f.read())
        f.close()
      except IOError:
        self.send_error(404,'File Not Found: %s' % self.path)

  #Handler for the POST requests
  def do_POST(self):
    form = cgi.FieldStorage(
      fp=self.rfile, 
      headers=self.headers,
      environ={'REQUEST_METHOD':'POST',
        'CONTENT_TYPE':self.headers['Content-Type'],
      })

    if self.path=="/getpose":
      if DBG_LOG: print("Data type is: %s" % form["data_type"].value)
      data_path = form["data_path"].value
      res = self.inference_request(data_path)

      # send response
      self.send_response(200)
      self.send_header('Content-type', 'application/json')
      self.end_headers()
      self.wfile.write(json.dumps(res).encode())
      return

  def log_message(self, format, *args):
    # disable logging
    if DBG_LOG: super(MyHandler, self).log_message(format, *args)

def main():
  _prepare_model()
  #infer(None, None, model)

  try:
    #Create a web server and define the handler to manage the
    #incoming request
    server = HTTPServer(('127.0.0.1', PORT_NUMBER), MyHandler)
    print('Listening on port, ' , PORT_NUMBER)
    print('Doc root ', DOC_ROOT)

    #Wait forever for incoming htto requests
    server.serve_forever()
  except KeyboardInterrupt:
    print('^C received, shutting down the web server')
    server.socket.close()

from multiprocessing import Process, freeze_support
if __name__ == "__main__":
  # Add support for when a program which uses multiprocessing has been frozen 
  # to produce a Windows executable. 
  freeze_support()
  #Process(target=main).start()
  main()
