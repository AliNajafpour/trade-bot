"""Authenticated status and explicit human controls; never submits an order."""
import argparse
import base64
from datetime import datetime, timezone
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import tempfile
import time
from urllib.parse import urlsplit

from .locking import atomic_replace, exclusive

ROOT = Path(__file__).resolve().parents[1]


class Controls:
    def __init__(self, config_path, username, password):
        if not username or ':' in username or len(password) < 20:
            raise ValueError('Dashboard user and password of at least 20 characters required')
        self.path = Path(config_path).resolve()
        self.username, self.password = username, password
        self.secret = secrets.token_bytes(32)

    def config(self):
        return json.loads(self.path.read_text(encoding='utf-8-sig'))

    def status(self):
        config = self.config()
        root = Path(config['run_dir'])
        if not root.is_absolute():
            root = ROOT / root
        try:
            source = json.loads((root / 'status.json').read_text(encoding='utf-8-sig'))
        except (FileNotFoundError, json.JSONDecodeError):
            source = {'status': 'starting'}
        # Only runner-owned public fields cross this authenticated boundary.
        allowed = ('status', 'updated_at', 'execution_ready', 'account_binding', 'positions',
                   'recent_orders', 'last_scan', 'monitor', 'error_type','adaptive_learning')
        result = {key: source[key] for key in allowed if key in source}
        account = source.get('account') or {}
        result['account'] = {key: account[key] for key in ('observed_at', 'balances', 'fees') if key in account}
        result['enabled'] = config.get('enabled') is True
        result['decision_mode']=config.get('decision_mode','council')
        result['max_capital_usdt'] = config.get('max_capital_usdt')
        result['stop_capabilities_verified'] = config.get('stop_capabilities_verified') is True
        result['user_action'] = config.get('user_action')
        return result

    def token(self):
        value = f'{int(time.time())}.{secrets.token_hex(16)}'
        return value + '.' + hmac.new(self.secret, value.encode(), hashlib.sha256).hexdigest()

    def valid_token(self, token):
        try:
            stamp, nonce, signature = token.split('.')
            value = f'{stamp}.{nonce}'
            expected = hmac.new(self.secret, value.encode(), hashlib.sha256).hexdigest()
            return 0 <= time.time() - int(stamp) <= 3600 and hmac.compare_digest(expected, signature)
        except (AttributeError, ValueError):
            return False

    def update(self, payload):
        if not isinstance(payload, dict) or payload.get('action') not in ('activate', 'deactivate'):
            raise ValueError('Unknown control action')
        with exclusive(self.path.with_suffix('.control.lock'), timeout=2):
            config = self.config()
            if payload['action'] == 'activate':
                from .live_runner import validate_activation
                state = self.status()
                binding = state.get('account_binding')
                if not binding or payload.get('account_binding') != binding:
                    raise ValueError('Current observed account binding must be confirmed')
                if config.get('account_binding') not in (None, binding):
                    raise ValueError('Configured and observed account bindings differ')
                try:
                    observed = datetime.fromisoformat(state['account']['observed_at'].replace('Z', '+00:00'))
                    fresh = 0 <= (datetime.now(timezone.utc) - observed).total_seconds() <= 60
                except (KeyError, ValueError, TypeError):
                    fresh = False
                if not fresh:
                    raise ValueError('Refresh the authenticated account snapshot before activation')
                if payload.get('native_stop_attested') is not True or payload.get('confirm_real_trading') is not True:
                    raise ValueError('Explicit real-trading and venue-tested native-stop attestations required')
                capital = payload.get('max_capital_usdt')
                if isinstance(capital, bool) or not isinstance(capital, (int, float)):
                    raise ValueError('Capital must be a positive finite USDT number')
                config.update(enabled=True, account_binding=binding, max_capital_usdt=capital,
                              stop_capabilities_verified=True)
                validate_activation(config, binding)
            else:
                config['enabled'] = False
            config['user_action'] = dict(at=datetime.now(timezone.utc).isoformat(),
                                         action=payload['action'], actor='dashboard_user')
            handle, temporary = tempfile.mkstemp(prefix='.control-', suffix='.json', dir=self.path.parent)
            try:
                with os.fdopen(handle, 'w', encoding='utf-8') as stream:
                    json.dump(config, stream, ensure_ascii=False, indent=2, allow_nan=False)
                    stream.flush()
                    os.fsync(stream.fileno())
                atomic_replace(temporary, self.path)
            finally:
                Path(temporary).unlink(missing_ok=True)
        return {'ok': True, 'enabled': config['enabled'], 'user_action': config['user_action']}


def server(config_path, host, port, username, password):
    controls = Controls(config_path, username, password)

    class Handler(BaseHTTPRequestHandler):
        server_version = 'Wallex'

        def setup(self):
            super().setup()
            self.connection.settimeout(10)

        def log_message(self, *args):
            pass  # Account paths, credentials and submitted values must not enter HTTP logs.

        def read_body(self):
            if not hasattr(self, '_body'):
                size = int(self.headers.get('Content-Length', '0'))
                if not 0 < size <= 4096 or self.headers.get('Transfer-Encoding'):
                    raise ValueError('A bounded JSON body is required')
                self._body = self.rfile.read(size)
                if len(self._body) != size:
                    raise ValueError('Incomplete request body')
            return self._body

        def reply(self, code, body, content_type='application/json; charset=utf-8', extra=None):
            # Closing with unread request bytes can reset TCP before Windows receives a denial.
            if self.command == 'POST':
                try:
                    self.read_body()
                except (ValueError, OSError):
                    pass  # Never drain an unbounded or stalled unauthenticated request.
            data = body.encode('utf-8') if isinstance(body, str) else json.dumps(body, ensure_ascii=False, allow_nan=False).encode('utf-8')
            self.send_response(code)
            for key, value in {'Content-Type': content_type, 'Content-Length': str(len(data)),
                               'Cache-Control': 'no-store', 'X-Content-Type-Options': 'nosniff',
                               'X-Frame-Options': 'DENY', 'Referrer-Policy': 'no-referrer',
                               'Content-Security-Policy': "default-src 'self'; script-src 'self' 'unsafe-inline'; style-src 'self' 'unsafe-inline'; frame-ancestors 'none'; base-uri 'none'; form-action 'self'",
                               **(extra or {})}.items():
                self.send_header(key, value)
            self.end_headers()
            self.wfile.write(data)

        def authenticated(self):
            try:
                scheme, encoded = self.headers.get('Authorization', '').split(' ', 1)
                user, password = base64.b64decode(encoded, validate=True).decode('utf-8').split(':', 1)
                valid = scheme.lower() == 'basic' and hmac.compare_digest(user.encode(), controls.username.encode()) and hmac.compare_digest(password.encode(), controls.password.encode())
            except (ValueError, UnicodeError):
                valid = False
            if not valid:
                self.reply(401, {'error': 'Authentication required'}, extra={'WWW-Authenticate': 'Basic realm="Wallex", charset="UTF-8"'})
            return valid

        def do_GET(self):
            path = urlsplit(self.path).path
            if path == '/health':
                self.reply(200, {'status': 'ok'})
                return
            if not self.authenticated():
                return
            try:
                if path == '/api/status':
                    self.reply(200, controls.status())
                elif path in ('/', '/control'):
                    page = (ROOT / 'wallex_lab/static/live-control.html').read_text(encoding='utf-8')
                    self.reply(200, page.replace('__CSRF_TOKEN__', controls.token()), 'text/html; charset=utf-8')
                else:
                    self.reply(404, {'error': 'Not found'})
            except (OSError, ValueError, KeyError):
                self.reply(503, {'error': 'Status unavailable; execution controls remain unchanged'})

        def do_POST(self):
            if not self.authenticated():
                return
            origin = urlsplit(self.headers.get('Origin', ''))
            if origin.scheme not in ('https', 'http') or origin.netloc != self.headers.get('Host') or not controls.valid_token(self.headers.get('X-CSRF-Token', '')):
                self.reply(403, {'error': 'Origin or CSRF token invalid; reload this page'})
                return
            if urlsplit(self.path).path != '/api/control':
                self.reply(404, {'error': 'Not found'})
                return
            try:
                if self.headers.get_content_type() != 'application/json':
                    raise ValueError('A bounded JSON body is required')
                payload = json.loads(self.read_body(), parse_constant=lambda value: (_ for _ in ()).throw(ValueError('Finite numbers required')))
                self.reply(200, controls.update(payload))
            except (ValueError, KeyError, TypeError, PermissionError) as error:
                self.reply(400, {'error': str(error)[:250]})
            except OSError:
                self.reply(503, {'error': 'Control write failed; verify status before retrying'})

    instance = ThreadingHTTPServer((host, port), Handler)
    instance.daemon_threads = True
    return instance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=8080)
    args = parser.parse_args()
    instance = server(args.config, args.host, args.port, os.getenv('WALLEX_DASHBOARD_USER', 'admin'), os.getenv('WALLEX_DASHBOARD_PASSWORD', ''))
    try:
        instance.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        instance.server_close()


if __name__ == '__main__':
    main()
