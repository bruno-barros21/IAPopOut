import json

with open('IAPopOut.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

with open('debug_nb.txt', 'w', encoding='utf-8') as out_f:
    out_f.write("--- Last 3 cells ---\n")
    for cell in nb['cells'][-3:]:
        out_f.write("Source:\n" + ''.join(cell.get('source', [])) + "\n")
        if 'outputs' in cell:
            for out in cell['outputs']:
                if out.get('output_type') == 'error':
                    out_f.write(f"ERROR: {out.get('ename')} - {out.get('evalue')}\n")
                    if 'traceback' in out:
                        out_f.write("TRACEBACK:\n" + "".join(out['traceback'][-5:]) + "\n")
                elif out.get('output_type') == 'stream':
                    out_f.write("STREAM: " + "".join(out.get('text'))[:200] + "\n")
        out_f.write("\n------------------\n")
