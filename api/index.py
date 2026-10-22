import sys
import os

# Add project root directory to sys.path so run.py and src can be imported
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from run import app


class PrefixMiddleware:
    """
    WSGI middleware that normalizes incoming paths rewritten by Vercel.
    Ensures routes like /api/index, /api/index/analyze, etc. map seamlessly to Flask routes.
    """
    def __init__(self, wsgi_app):
        self.wsgi_app = wsgi_app

    def __call__(self, environ, start_response):
        path = environ.get('PATH_INFO', '')
        if path.startswith('/api/index'):
            environ['PATH_INFO'] = path[len('/api/index'):] or '/'
        elif path.startswith('/api') and not (path.startswith('/api/samples') or path.startswith('/api/presets') or path.startswith('/api/benchmark')):
            environ['PATH_INFO'] = path[len('/api'):] or '/'
        return self.wsgi_app(environ, start_response)


app.wsgi_app = PrefixMiddleware(app.wsgi_app)
