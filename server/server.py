#!/usr/bin/python
from BaseHTTPServer import BaseHTTPRequestHandler,HTTPServer
from os import curdir, sep, path
import cgi

import json

DOC_ROOT = './public'
PORT_NUMBER = 8080

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
class myHandler(BaseHTTPRequestHandler):
	#Handler for the GET requests
  def do_GET(self):
    if self.path=="/":
      self.path="/index.html"

    #Check the file extension required and
    #set the right mime type
    msg = None
    if self.path.endswith(".html"):
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
        msg = 'released...'
      else:
        return # invalid request

    if msg:
      self.send_response(200)
      self.send_header('Content-type', 'text')
      self.end_headers()
      self.wfile.write(msg)
    else:
      # serve file
      try:
        #Open the static file requested and send it
        f = open(DOC_ROOT + self.path) 
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
      print("Data type is: %s" % form["data_type"].value)
      data_path = form["data_path"].value
      _prepare_model()
      all_preds = infer(path.dirname(data_path), data_path, model)

      # send response
      self.send_response(200)
      self.send_header('Content-type', 'application/json')
      self.end_headers()
      self.wfile.write(json.dumps({
        'status': 200,
        'message': "request with %s!" % form["data_type"].value,
        'result': all_preds.tolist()
      }))
      return

_prepare_model()

try:
  DOC_ROOT = path.join(path.dirname(path.realpath(__file__)), DOC_ROOT)
  print('Doc root ', DOC_ROOT)
  #Create a web server and define the handler to manage the
  #incoming request
  server = HTTPServer(('', PORT_NUMBER), myHandler)
  print('Listening on port ' , PORT_NUMBER)

  #Wait forever for incoming htto requests
  server.serve_forever()

except KeyboardInterrupt:
	print('^C received, shutting down the web server')
	server.socket.close()
