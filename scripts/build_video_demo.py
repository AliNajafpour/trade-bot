"""Portable recording kit, using the real dashboard with synthetic demo inputs."""
from pathlib import Path
import json
import re
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from wallex_lab.council_dashboard import weekly_dashboard_html

def main():
    out=ROOT/'video-demo';out.mkdir(exist_ok=True)
    html=weekly_dashboard_html()
    runtime=(ROOT/'wallex_lab/static/video-demo-runtime.js').read_text(encoding='utf-8')
    html=html[:html.index('let pending=false;async function update()')]+runtime+'\n</script></body></html>'
    html=html.replace('<span class="badge warn">معاملات کاغذی</span>','').replace('<span class="paper-label">PAPER ACCOUNT</span>','').replace('حساب مستقل · اسپات','اسپات')
    html=html.replace('http://127.0.0.1:8832/','architecture.html').replace('مقایسهٔ نسخه‌ها ↗','معماری و فلوچارت‌ها ↗').replace('href="/status.json"','href="architecture.html"').replace('دادهٔ کامل ↗','روش‌ها و الگوریتم‌ها ↗')
    controls='<div class="demo-controls"><span id="demo-timer" class="number">00:00 / 05:00</span><button id="demo-pause">مکث</button><button id="demo-reset">شروع دوباره</button><button id="demo-speed">1×</button><input id="demo-seek" type="range" min="0" max="300" step="1" value="0" aria-label="زمان دمو"><a href="architecture.html">۱۰ نمودار و فلوچارت ↗</a></div>'
    html=html.replace('<main>','<main>'+controls,1)
    chart='<section class="section"><div class="section-head"><h2>نمودار قیمت</h2><div class="demo-assets"><button data-asset="BTC/USDT">BTC</button><button data-asset="ETH/USDT">ETH</button><button data-asset="SOL/USDT">SOL</button><button data-asset="S/USDT">S</button></div></div><div class="card card-pad"><div class="demo-chart-title"><b id="demo-asset" class="en">ETH/USDT</b><span id="demo-price" class="number"></span><small>خط سبز: TP · خط قرمز: SL</small></div><canvas id="demo-chart" aria-label="نمودار قیمت شبیه‌سازی‌شده"></canvas></div></section>'
    html=html.replace('<section id="signals"',chart+'<section id="signals"',1)
    css='.demo-controls #demo-timer{direction:ltr;unicode-bidi:isolate}.demo-controls{display:flex;gap:12px;align-items:center;flex-wrap:wrap;padding:14px 0 22px}.demo-controls input{min-width:100px;flex:1;accent-color:var(--cyan)}.demo-controls button,.demo-assets button{background:var(--panel);border:1px solid var(--line);border-radius:7px;padding:8px 14px;color:var(--text);font:inherit;cursor:pointer}.demo-controls a{color:var(--cyan);font-size:11px}.demo-controls button:focus-visible,.demo-assets button:focus-visible{outline:2px solid var(--cyan)}.demo-assets{display:flex;gap:8px;direction:ltr}.demo-chart-title{display:flex;align-items:center;gap:15px}.demo-chart-title small{margin-right:auto;color:var(--muted)}#demo-chart{width:100%;height:240px;display:block;margin-top:10px}.demo-controls .badge{font-size:10px}@media(max-width:700px){.demo-controls input{flex-basis:100%}}'
    html=html.replace('</style>',css+'</style>',1)
    html=re.sub(r'@import[^;]+;','',html)
    assert 'fetch(' not in html and '/status.json' not in html and '127.0.0.1' not in html
    (out/'dashboard.html').write_text(html,encoding='utf-8')
    (out/'architecture.html').write_text((ROOT/'wallex_lab/static/video-architecture.html').read_text(encoding='utf-8'),encoding='utf-8')
    (out/'README.txt').write_text('dashboard.html را با Chrome یا Edge باز کنید. دمو پنج دقیقه خودکار اجرا می‌شود؛ Space مکث و F11 تمام‌صفحه. همهٔ قیمت‌ها، خبرها و معاملات این صفحه ساختگی و برای نمایش رابط‌اند.\narchitecture.html شامل ۱۰ نمودار و فلوچارت است؛ کلیدهای جهت‌دار و دکمه‌های پایین صفحه را بزنید.\nهر دو HTML مستقل، بدون نصب، اینترنت، سرور یا API key باز می‌شوند. ZIP را برای دوستتان بفرستید و پس از Extract فایل‌ها را باز کنید.\nمتن معرفی و آمار بک‌تست واقعی در narration.txt است. اطلاعات حساب واقعی در بسته نیست.\n',encoding='utf-8')
    with zipfile.ZipFile(out/'wallex-video-kit.zip','w',zipfile.ZIP_DEFLATED) as archive:
        for name in ('dashboard.html','architecture.html','README.txt','narration.txt'):
            if (out/name).exists():archive.write(out/name,arcname=name)
    print(json.dumps(dict(output=str(out),offline=True,synthetic_demo_seconds=300,uses_existing_dashboard=True)))

if __name__=='__main__':main()
