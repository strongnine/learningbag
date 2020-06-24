import tornado.ioloop
import tornado.web

class MainHandler(tornado.web.RedirectHandler):
    def get(self):
        self.write("Hello, world")

def make_app():
    return tornado.web.Application([
        (r"/", MainHandler),
    ])

if __name__ == "__main__":
    app = make_app()
    app.listen(8888)
    tornado.ioloop.IOLoop.current().start()

# from flask import Flask
# app = Flask(__name__)
#
# @app.route("/")
# def hello():
#     return "Hello World!"
#
# if __name__ == "__main__":
#     app.run()

# import tesserocr
# from PIL import Image
# image = Image.open('762408.jpg')
# print(tesserocr.image_to_text(image))