"""Checkpointed LangGraph specialist execution with the existing research contracts."""
from copy import deepcopy
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import TypedDict

from langgraph.checkpoint.sqlite import SqliteSaver
from langgraph.graph import END, START, StateGraph

from . import llm, specialists
from .locking import exclusive

ROLES = ('regime', 'sentiment', 'fundamentals')


class AnalysisState(TypedDict, total=False):
    request: dict
    prepared: dict
    regime_outcome: dict
    sentiment_outcome: dict
    fundamentals_outcome: dict
    result: dict


def _json(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(',', ':'))


def _digest(value):
    return hashlib.sha256(_json(value).encode('utf-8')).hexdigest()


def _cached_provider(provider, directory):
    """Keep completed backend receipts even if a node's checkpoint commit is interrupted."""
    def cached(packet, prompt, schema, output_root, **kwargs):
        role = Path(output_root).name
        key = _digest(dict(packet=packet, prompt=prompt, schema=schema, options=kwargs))
        path = directory / role / (key + '.json')
        if path.exists():
            record = json.loads(path.read_text(encoding='utf-8'))
            if record['request_sha256'] != key or record['receipt_sha256'] != _digest(record['receipt']):
                raise ValueError('Graph receipt cache failed integrity check')
            return dict(record['receipt'], cache_hit=True, cli_calls=0, api_calls=0)
        receipt = provider(packet, prompt, schema, output_root, **kwargs)
        if receipt.get('status') != 'ok':
            return receipt
        try:
            llm.validate_value(receipt['response'], schema)
            if role in ('regime', 'decision'):
                specialists._validate_semantics(receipt['response'])
        except (ValueError, TypeError, KeyError):
            return receipt
        record = dict(request_sha256=key, receipt=receipt, receipt_sha256=_digest(receipt))
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix('.tmp')
        temporary.write_text(_json(record), encoding='utf-8')
        temporary.replace(path)
        return receipt
    return cached


def build_graph(checkpointer, provider, run_dir):
    """Prepare, fan out three independent roles, then join at the final decision."""
    def prepare(state):
        return {'prepared': specialists.prepare_analysis(**state['request'])}

    def role_node(state, role):
        job = state['prepared']['jobs'].get(role)
        return {role + '_outcome': specialists.infer_role(job, role, run_dir, provider) if job else {}}

    def decision(state):
        prepared = state['prepared']
        result = deepcopy(prepared['result'])
        for role in ROLES:
            specialists.merge_role(result, role, state[role + '_outcome'])
        return {'result': specialists.decide_analysis(prepared, result, run_dir, provider)}

    builder = StateGraph(AnalysisState)
    builder.add_node('prepare', prepare)
    builder.add_edge(START, 'prepare')
    for role in ROLES:
        builder.add_node(role, partial(role_node, role=role))
        builder.add_edge('prepare', role)
    builder.add_node('decision', decision)
    builder.add_edge(list(ROLES), 'decision')
    builder.add_edge('decision', END)
    return builder.compile(checkpointer=checkpointer)


def analyze(summary, context, symbol, asof, run_dir, provider=None):
    """Orchestrator-compatible analyzer; repeated identical requests resume SQLite state."""
    prepared = specialists.prepare_analysis(summary, context, symbol, asof)
    if provider is None:
        from .providers import provider_from_env
        provider = provider_from_env()
    provider_identity = getattr(provider, 'cache_identity', None) or {
        'module': getattr(provider, '__module__', type(provider).__module__),
        'name': getattr(provider, '__qualname__', type(provider).__qualname__)}
    implementation = hashlib.sha256(Path(__file__).read_bytes() + Path(specialists.__file__).read_bytes()).hexdigest()
    thread_id = _digest(dict(prepared=prepared, provider=provider_identity, implementation=implementation))
    directory = Path(run_dir).resolve()
    directory.mkdir(parents=True, exist_ok=True)
    checkpoint_path = directory / 'graph_checkpoints.sqlite'
    receipt_directory = directory / 'graph_receipts' / thread_id
    cached = _cached_provider(provider, receipt_directory)
    # ponytail: one writer per run directory; split locks by thread if independent runs share it.
    with exclusive(directory / 'graph.lock'), SqliteSaver.from_conn_string(str(checkpoint_path)) as checkpointer:
        attempt_root = directory / 'graph_attempts' / thread_id
        graph = build_graph(checkpointer, cached, attempt_root / 'initial')
        config = {'configurable': {'thread_id': thread_id}, 'max_concurrency': 3}
        snapshot = graph.get_state(config)
        resumed = bool(snapshot.values)
        complete = resumed and not snapshot.next and 'result' in snapshot.values
        if complete and snapshot.values['result']['status'] == 'ok':
            state = snapshot.values
        else:
            if complete:
                # A fresh output root retries backend errors whose original receipts remain cached.
                attempt = snapshot.config['configurable']['checkpoint_id']
                graph = build_graph(checkpointer, cached, attempt_root / attempt)
            request = dict(summary=summary, context=context, symbol=prepared['result']['symbol'],
                           asof=prepared['result']['asof'])
            state = graph.invoke(None if resumed and not complete else {'request': request}, config)
        result = deepcopy(state['result'])
        if complete and snapshot.values['result']['status'] == 'ok':
            for receipt in result['receipts']:
                receipt.update(cache_hit=True, cli_calls=0, api_calls=0)
    result['graph'] = dict(engine='langgraph', thread_id=thread_id, checkpoint_path=str(checkpoint_path),
        resumed=resumed, nodes=['prepare', *ROLES, 'decision'])
    return result
