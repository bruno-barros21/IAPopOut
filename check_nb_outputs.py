import json

with open('IAPopOut.ipynb', 'r', encoding='utf-8') as f:
    nb = json.load(f)

for i, cell in enumerate(nb['cells']):
    if cell['cell_type'] == 'code' and cell.get('outputs'):
        print(f"--- Cell {i} Outputs ---")
        for out in cell['outputs']:
            if out.get('output_type') == 'error':
                print("Error Name:", out.get('ename'))
                print("Error Value:", out.get('evalue'))
                if 'traceback' in out:
                    # just print the last few lines of traceback
                    print("Traceback:", "".join(out['traceback'][-5:]).strip())
            elif out.get('output_type') == 'stream':
                print("Stream:", "".join(out.get('text'))[:200].strip())
        print()
