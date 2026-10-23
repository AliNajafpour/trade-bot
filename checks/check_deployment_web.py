"""Offline HTTP control checks. No key, account connection or exchange write."""
import base64
from datetime import datetime, timezone
from http.client import HTTPConnection
import json
from pathlib import Path
import re
import sys
from tempfile import TemporaryDirectory
from threading import Thread

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from wallex_lab.deployment_web import Controls, server
from wallex_lab.live_execution import DEFAULTS


def main():
    with TemporaryDirectory() as directory:
        root = Path(directory)
        config_path = root / 'control.json'
        config = dict(DEFAULTS, run_dir=str(root), max_capital_usdt=None)
        config_path.write_text(json.dumps(config), encoding='utf-8')
        state = dict(account_binding='offline-binding', updated_at=datetime.now(timezone.utc).isoformat(),
                     status='ready_disabled', account={'observed_at': datetime.now(timezone.utc).isoformat(),
                     'balances': {'USDT': {'available': '100', 'total': '100'}}, 'API_KEY': 'should-never-leak'},
                     positions=[], API_KEY='should-never-leak')
        (root / 'status.json').write_text(json.dumps(state), encoding='utf-8')
        try:
            Controls(config_path, 'admin', 'weak')
        except ValueError:
            pass
        else:
            raise AssertionError('Weak password accepted')
        app = server(config_path, '127.0.0.1', 0, 'admin', 'offline-only-strong-password')
        thread = Thread(target=app.serve_forever, daemon=True)
        thread.start()
        host = f'127.0.0.1:{app.server_port}'
        authorization = 'Basic ' + base64.b64encode(b'admin:offline-only-strong-password').decode()

        def request(method, path, payload=None, auth=True, token=None, origin=None):
            connection = HTTPConnection('127.0.0.1', app.server_port, timeout=5)
            headers = {'Content-Type': 'application/json'}
            if auth:
                headers['Authorization'] = authorization
            if token:
                headers['X-CSRF-Token'] = token
            if origin:
                headers['Origin'] = origin
            body = json.dumps(payload) if payload is not None else None
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            code, data, response_headers = response.status, response.read().decode(), dict(response.getheaders())
            connection.close()
            return code, data, response_headers

        try:
            assert request('GET', '/health', auth=False)[0:2] == (200, '{"status": "ok"}')
            for path in ('/', '/control', '/api/status', '/anything'):
                assert request('GET', path, auth=False)[0] == 401
            assert request('GET', '/api/status')[0] == 200
            assert 'should-never-leak' not in request('GET', '/api/status')[1]
            assert json.loads(config_path.read_text())['enabled'] is False
            code, page, headers = request('GET', '/control')
            assert code == 200 and headers['Cache-Control'] == 'no-store'
            assert headers['X-Frame-Options'] == 'DENY'
            token = re.search("const csrf='([^']+)'", page)[1]
            activation = dict(action='activate', max_capital_usdt=25,
                              account_binding='offline-binding', native_stop_attested=True,
                              confirm_real_trading=True)
            # Rejected POSTs must drain bounded bodies so Windows receives the HTTP denial.
            for _ in range(25):
                assert request('POST', '/api/control', activation, auth=False)[0] == 401
                assert request('POST', '/api/control', activation, token='bad', origin='http://' + host)[0] == 403
                assert request('POST', '/not-found', activation, token=token, origin='http://' + host)[0] == 404
            assert request('POST', '/api/control', activation, token=token, origin='https://evil.example')[0] == 403
            assert request('POST', '/api/control', activation, token=token)[0] == 403
            assert request('POST', '/api/control', activation, token='bad', origin='http://' + host)[0] == 403
            for bad in ({'max_capital_usdt': 0}, {'max_capital_usdt': -10}, {'max_capital_usdt': True},
                        {'max_capital_usdt': float('inf')}, {'account_binding': 'wrong'},
                        {'native_stop_attested': False}, {'confirm_real_trading': False}):
                code, result, _ = request('POST', '/api/control', dict(activation, **bad), token=token, origin='http://' + host)
                assert code == 400, (bad, code, result)
                assert json.loads(config_path.read_text())['enabled'] is False
            state['account']['observed_at'] = '2000-01-01T00:00:00+00:00'
            (root / 'status.json').write_text(json.dumps(state), encoding='utf-8')
            assert request('POST', '/api/control', activation, token=token, origin='http://' + host)[0] == 400
            state['account']['observed_at'] = datetime.now(timezone.utc).isoformat()
            (root / 'status.json').write_text(json.dumps(state), encoding='utf-8')
            assert request('POST', '/api/control', activation, token=token, origin='http://' + host)[0] == 200
            saved = json.loads(config_path.read_text())
            assert saved['enabled'] is True and saved['stop_capabilities_verified'] is True
            assert saved['max_capital_usdt'] == 25
            assert saved['user_action']['actor'] == 'dashboard_user'
            assert request('POST', '/api/control', {'action': 'deactivate'}, token=token, origin='http://' + host)[0] == 200
            assert json.loads(config_path.read_text())['enabled'] is False
            assert json.loads(config_path.read_text())['user_action']['action'] == 'deactivate'
        finally:
            app.shutdown()
            app.server_close()
            thread.join(timeout=5)
    print('PASS: authenticated controls, CSRF/origin, fresh binding, native-stop attestation, bounded capital, user-only activation, deactivation; no exchange access')


if __name__ == '__main__':
    main()
