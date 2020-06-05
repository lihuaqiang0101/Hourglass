import logging
from concurrent.futures import ThreadPoolExecutor
from tornado.concurrent import run_on_executor
import tornado.httpserver
import tornado.ioloop
import tornado.options
import tornado.web
import tornado.gen
import json
import traceback
import pigweights
import base64
import re
import time
# from pigweights import Pigweights
from Net import HourglassNet

# model = Pigweights()
class MainHandler(tornado.web.RequestHandler):
    executor = ThreadPoolExecutor(32)

    @tornado.gen.coroutine
    def get(self):
        '''get接口'''
        htmlStr = '''
                    <!DOCTYPE HTML><html>
                    <meta charset="utf-8">
                    <head><title>Get page</title></head>
                    <body>
                    <form		action="/post"	method="post" >
                    imageBase64:<br>
                    <input type="text"      name ="imageBase64"     /><br>
                    fileId:<br>
                    <input type="text"      name ="fileId"     /><br>
                    isFromAlbum:<br>
                    <input type="text"      name ="isFromAlbum"     /><br>
                    
                    <input type="submit"	value="test"	/>
                    </form></body> </html>
                '''
        self.write(htmlStr)

    @tornado.gen.coroutine
    def post(self):
        '''post接口， 获取参数'''
        imageBase64 = self.get_argument("imageBase64", None)
        fileId = self.get_argument("fileId", None)
        isFromAlbum = self.get_argument("isFromAlbum", None)
        yield self.coreOperation(imageBase64, fileId, isFromAlbum)

    @run_on_executor
    def coreOperation(self, imageBase64, fileId, isFromAlbum):
        '''主函数'''
        global t
        try:
            # global result
            data = imageBase64.split(',')[-1]
            if re.match('^([A-Za-z0-9+/]{4})*([A-Za-z0-9+/]{4}|[A-Za-z0-9+/]{3}=|[A-Za-z0-9+/]{2}==)$', data):
                if(len(data)%3 == 1):
                    data += '=='
                elif(len(data)%3 == 2):
                    data += '='

                data = base64.b64decode(data)
                path = 'D://test/' + str(fileId)+ '.jpg'
                with open(path, 'wb') as f:
                    f.write(data)

                nowtime = time.time()
                result = pigweights.pigweights(path, isFromAlbum)  # 可调用其他接口
                # result = model.pigweights(path, isFromAlbum)  # 可调用其他接口
                aftertime = time.time()
                t = aftertime - nowtime
                if isinstance(result, float):
                    result = json.dumps({'code': 200, 'msg': 'successful', 'weight': result, 'fileId': fileId, 'isFromAlbum': isFromAlbum})
                elif isinstance(result, str):
                    result = json.dumps({'code': 4001, 'msg': 'its not a pig', 'fileId': fileId, 'isFromAlbum': isFromAlbum})

            else:
                result = json.dumps({'code': 500, 'msg': 'wrong input imageBase64', 'fileId': fileId, })
            self.write(result)
        except Exception:
            print('traceback.format_exc():\n%s' % traceback.format_exc())
            result = json.dumps({'code': 500, 'msg': '系统异常', 'weight': 0, 'fileId': fileId, 'isFromAlbum': isFromAlbum})
            self.write(result)
        logging.info(result + ' use ' + str(round(t*1000, 2)) + 'ms')



if __name__ == "__main__":
    tornado.options.parse_command_line()
    app = tornado.web.Application(handlers=[(r'/post', MainHandler)], autoreload=False, debug=False)
    http_server = tornado.httpserver.HTTPServer(app)
    http_server.listen(8801, '0.0.0.0')
    tornado.ioloop.IOLoop.instance().start()