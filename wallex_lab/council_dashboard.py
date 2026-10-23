"""Persian council instrument board. Read-only, with evidence before trade status.

Navy/steel panels, amber admission gates, cyan observations. Native browser
controls keep the live screen dependency-free and keyboard accessible.
"""
from pathlib import Path


def dashboard_html():
    return (Path(__file__).with_name('static') / 'council-dashboard.html').read_text(encoding='utf-8')


def weekly_dashboard_html():
    styles=dashboard_html().split('<style>',1)[1].split('</style>',1)[0]
    return (Path(__file__).with_name('static')/'survival-news-dashboard.html').read_text(encoding='utf-8').replace('/* SHARED_COUNCIL_STYLE */',styles)
