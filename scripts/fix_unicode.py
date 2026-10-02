"""Fix all non-ASCII characters in Python source files."""
from pathlib import Path

files = [
    'scripts/build_dataset.py',
    'ml/train.py',
    'run_pipeline.py',
    'cli/recommend.py',
]

for fp in files:
    p = Path(fp)
    try:
        raw = p.read_bytes()
        text = raw.decode('utf-8', errors='replace')
        # Replace common unicode symbols with ASCII equivalents
        text = text.replace('\u2192', '->')
        text = text.replace('\u2714', '[OK]')
        text = text.replace('\u26a0', '[!]')
        text = text.replace('\u26a1', '>>')
        text = text.replace('\u2588', '#')
        text = text.replace('\u2550', '=')
        text = text.replace('\u2500', '-')
        text = text.replace('\u2501', '-')
        text = text.replace('\u254c', '-')
        text = text.replace('\u2013', '-')
        text = text.replace('\u2014', '--')
        text = text.replace('\u2018', "'")
        text = text.replace('\u2019', "'")
        text = text.replace('\u201c', '"')
        text = text.replace('\u201d', '"')
        text = text.replace('\u25b6', '>')
        text = text.replace('\u2022', '*')
        text = text.replace('\u2248', '~=')
        text = text.replace('\u00b1', '+/-')
        text = text.replace('\u2713', 'OK')
        text = text.replace('\u2717', 'FAIL')
        # Any remaining non-ASCII → '?'
        text = text.encode('ascii', errors='replace').decode('ascii')
        p.write_text(text, encoding='utf-8')
        print(f'Fixed: {fp}')
    except Exception as e:
        print(f'ERROR {fp}: {e}')
