"""Dash app entry point"""
import logging
import os

import dash
import flask
from dash import _dash_renderer

_dash_renderer._set_react_version('18.2.0')

from app.callbacks import register_callbacks
from app.data_loader import load_artifacts, missing_artefacts_message
from app.layout import build_layout, build_unavailable_layout

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s %(levelname)s %(name)s: %(message)s',
)
logger = logging.getLogger('app.main')

server = flask.Flask(__name__)
app = dash.Dash(
    __name__,
    server=server,
    suppress_callback_exceptions=True,
    title='Readmission dashboard',
    update_title=None,
)


@server.after_request
def _allow_iframe(response):
    response.headers['X-Frame-Options'] = 'ALLOWALL'
    response.headers['Content-Security-Policy'] = 'frame-ancestors *'
    return response


try:
    _bundle = load_artifacts()
    app.layout = build_layout(_bundle)
    register_callbacks(app, _bundle)
    logger.info('Dashboard ready — %d patients loaded.', _bundle.n_patients)
except FileNotFoundError as exc:
    logger.error('Artefact check failed: %s', exc)
    app.layout = build_unavailable_layout(missing_artefacts_message() or str(exc))


def main():
    host = os.environ.get('DASH_HOST', '0.0.0.0')
    port = int(os.environ.get('DASH_PORT', '8050'))
    debug = os.environ.get('DASH_DEBUG', '0') == '1'
    app.run(host=host, port=port, debug=debug)


if __name__ == '__main__':
    main()
