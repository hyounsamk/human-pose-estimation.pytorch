# coding=utf-8
#!/usr/bin/python
from http.server import BaseHTTPRequestHandler, HTTPServer
from os import curdir, sep, path
import sys
import cgi
import json
from multiprocessing import Process, freeze_support
from urllib.parse import urlparse, parse_qs

DOC_ROOT = path.join(path.dirname(path.realpath(__file__)), 'public')
PORT_NUMBER = 8080
DBG_LOG = False

class MyHandler(BaseHTTPRequestHandler):
	# Handler for the GET requests
  def do_GET(self):
    global DBG_LOG
    try:
      # /path/to?k1=v1&k2=v2#frag1
      parsed_url = urlparse(self.path)
      if DBG_LOG: print('>>> REQ: %s>>>%s,%s,%s<<<' %(self.path, parsed_url.path, parsed_url.query, parsed_url.fragment))
      result = {
        "status": 200,
        "path": parsed_url.path
      }
      queries = parse_qs(parsed_url.query)
      for key, value in queries.items():
        result[key] = value[0]

      self.send_response(200)
      self.send_header('Content-type', 'application/json')
      self.end_headers()
      self.wfile.write(json.dumps(result).encode())
      if self.path == '/': DBG_LOG = not DBG_LOG
    except Exception as e:
      print('!!! SERVER GET ERROR: ')
      print(e.args)
      self.send_error(500,'Error: %s' % self.path)

  # Handler for the POST requests
  def do_POST(self):
    try:
      form = cgi.FieldStorage(
        fp=self.rfile, 
        headers=self.headers,
        environ={'REQUEST_METHOD':'POST',
          'CONTENT_TYPE':self.headers['Content-Type'],
        })

      if DBG_LOG: print('>>> REQ: %s<<<' % self.path)
      result = {
        "status": 200,
        "message": 'your request'
      }
      for key in form.keys():
        result[key] = form.getvalue(key)

      # send response
      self.send_response(200)
      self.send_header('Content-type', 'application/json')
      self.end_headers()
      self.wfile.write(json.dumps(result).encode())
    except Exception as e:
      print('!!! SERVER POST ERROR: ')
      print(e.args)
      self.send_error(500,'Error: %s' % self.path)

  def log_message(self, format, *args):
    # disable logging
    if DBG_LOG: super(MyHandler, self).log_message(format, *args)

def start_server():
  try:
    # Create a web server and define the handler to manage the incoming request
    server = HTTPServer(('localhost', PORT_NUMBER), MyHandler)
    print('Running on port ', PORT_NUMBER)
    print('Doc root ', DOC_ROOT)

    # Wait forever for incoming http requests
    server.serve_forever()
  except KeyboardInterrupt:
    print('^C received, shutting down the web server')
    server.socket.close()

from urllib import request, parse
import time

def get_request(base_url, get_count):
  # GET request
  times = []
  count = get_count
  failure = 0
  while True:
    if count <= 0:
      break
    count -= 1
    t_start = time.time()
    res = request.urlopen('%s%s?q=%.7f#%s' % (base_url, '/path/to/get', t_start, 'fragment'), timeout=2) # 2 seconds
    if res.getcode() != 200:
      failure += 1
      print('RES code: %d' % res.getcode())
    else:
      res = res.read()
      obj = json.loads(res.decode('utf-8'))
      if float(obj['q']) != t_start:
        failure += 1
        print('DIFF', obj['q'], t_start)
    #print(res)
    times.append(time.time() - t_start)
  return times, failure

def post_request(base_url, post_count):
  # POST request
  times = []
  count = post_count
  failure = 0
  while True:
    if count <= 0:
      break
    count -= 1

    t_start = time.time()
    data = { 'q': t_start }
    data = parse.urlencode(data).encode()
    headers = { "Content-type": "application/x-www-form-urlencoded", "Accept": 'application/json' }

    req = request.Request('%s%s' % (base_url, '/path/to/post'), data, headers)
    res = request.urlopen(req, timeout=2) # 2 seconds
    if res.getcode() != 200:
      failure += 1
      print('RES code: %d' % res.getcode())
    else:
      res = res.read()
      obj = json.loads(res.decode('utf-8'))
      if float(obj['q']) != t_start:
        failure += 1
        print('DIFF', obj['q'], t_start)
    #print(res)
    times.append(time.time() - t_start)
  return times, failure

def post_request2(base_url, post_count):
  # POST request
  times = []
  count = post_count
  failure = 0
  while True:
    if count <= 0:
      break
    count -= 1

    t_start = time.time()
    data = { 
      'data_type': 'path',
      'data_path': 'data/coco_simple/images2/person_keypoints.json'
      #'data_path': 'data/coco_simple/images2/person_keypoints_all.json'
    }
    data = parse.urlencode(data).encode()
    headers = { "Content-type": "application/x-www-form-urlencoded", "Accept": 'application/json' }

    req = request.Request('%s%s' % (base_url, '/getpose'), data, headers)
    res = request.urlopen(req, timeout=5) # 5 seconds
    if res.getcode() != 200:
      failure += 1
      print('RES code: %d' % res.getcode())
    else:
      res = res.read()
      obj = json.loads(res.decode('utf-8'))
      if float(obj['status']) != 200:
        failure += 1
        print('DIFF', obj['status'])
    #print(res)
    times.append(time.time() - t_start)
  return times, failure


def report_time(msg, times, failure):
  print('%s: failure: %d / %d' % (msg, failure, len(times)))
  if len(times) > 0:
    print('  max: %.7f' % max(times))
    print('  min: %.7f' % min(times))
    print('  avg: %.7f' % (sum(times) / len(times)))

def start_request(get_count, post_count):
  # base url은 'localhost'가 아닌 '127.0.0.1'로 명시한다.
  # 'localhost'로 명시하면 ip lookup을 위한 시간이 더 소요된다.
  base_url = 'http://127.0.0.1:%d' % PORT_NUMBER

  print('======================')
  print('Start request: GET: %d, POST: %d times' % (get_count, post_count))

  print('-----------------------')
  times, failure = get_request(base_url, get_count)
  report_time('Elapsed time for GET', times, failure)

  print('-----------------------')
  times, failure = post_request2(base_url, post_count)
  report_time('Elapsed time for POST', times, failure)

if __name__ == "__main__":
  # Add support for when a program which uses multiprocessing has been frozen 
  # to produce a Windows executable. 
  freeze_support()

  # start server
  if len(sys.argv) > 1 :
    Process(target=start_server).start()
    #start_server()
    time.sleep(2) # 2 seconds

  start_request(0, 1000)

  print('=====================')

