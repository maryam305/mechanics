import json
import glob

notebooks = glob.glob("*.ipynb")
for nb_file in notebooks:
    try:
        with open(nb_file, 'r', encoding='utf-8') as f:
            nb = json.load(f)
        code = []
        for cell in nb.get('cells', []):
            if cell['cell_type'] == 'code':
                code.append("".join(cell.get('source', [])))
        
        out_name = nb_file.replace('.ipynb', '.py')
        with open(out_name, 'w', encoding='utf-8') as f:
            f.write("\n\n".join(code))
        print(f"Extracted {nb_file} to {out_name}")
    except Exception as e:
        print(f"Failed to parse {nb_file}: {e}")
