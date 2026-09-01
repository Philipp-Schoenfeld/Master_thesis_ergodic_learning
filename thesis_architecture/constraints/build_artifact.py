#!/usr/bin/env python3
r"""
build_artifact.py
=================
Splice the exported results JSON into the artifact HTML template.

Kept as a build step rather than hand-pasting so the page can be regenerated
from fresh numbers: re-run `export_artifact_data.py`, re-run this, republish.

    python build_artifact.py --template t.html --data results/artifact_data.json --out page.html
"""
import argparse
import json
import os

_here = os.path.dirname(os.path.abspath(__file__))
TOKEN = '__DATA__'


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--template', required=True)
    p.add_argument('--data', default=os.path.join(_here, 'results', 'artifact_data.json'))
    p.add_argument('--out', required=True)
    args = p.parse_args()

    with open(args.template, encoding='utf-8') as f:
        html = f.read()
    if TOKEN not in html:
        raise SystemExit(f"template has no {TOKEN} placeholder")

    with open(args.data, encoding='utf-8') as f:
        payload = f.read()
    json.loads(payload)   # fail loudly here rather than in the browser

    # The payload rides inside <script type="application/json">, so the only
    # sequence that can break out of it is a literal closing script tag.
    payload = payload.replace('</', '<\\/')

    with open(args.out, 'w', encoding='utf-8') as f:
        f.write(html.replace(TOKEN, payload))

    mb = os.path.getsize(args.out) / 1e6
    print(f"Wrote {args.out}  ({mb:.2f} MB)")


if __name__ == '__main__':
    main()
