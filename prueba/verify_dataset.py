# verify_dataset.py
import os, sys, glob, yaml

def load_yaml(yaml_path):
    with open(yaml_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)

def check_split(base, split, nc):
    imgs = sorted(glob.glob(os.path.join(base, split, 'images', '*.*')))
    lbls_dir = os.path.join(base, split, 'labels')
    problems = []
    n_ok = 0
    for img in imgs:
        name = os.path.splitext(os.path.basename(img))[0]
        lbl = os.path.join(lbls_dir, name + '.txt')
        if not os.path.exists(lbl):
            problems.append(f'[MISSING] {split}: no label for {name}')
            continue
        with open(lbl, 'r', encoding='utf-8') as f:
            lines = [ln.strip() for ln in f if ln.strip()]
        if not lines:
            problems.append(f'[EMPTY] {split}: empty label {name}.txt')
            continue
        ok_line = True
        for ln in lines:
            parts = ln.split()
            if len(parts) != 5:
                problems.append(f'[FORMAT] {split}:{name}.txt -> \"{ln}\" (expected 5 vals)')
                ok_line = False
                continue
            try:
                cid = int(parts[0]); x,y,w,h = map(float, parts[1:])
            except Exception:
                problems.append(f'[PARSE]  {split}:{name}.txt -> \"{ln}\"')
                ok_line = False
                continue
            if not (0 <= cid < nc):
                problems.append(f'[CLASS]  {split}:{name}.txt -> class {cid} not in [0,{nc-1}]')
                ok_line = False
            if not (0<=x<=1 and 0<=y<=1 and 0<=w<=1 and 0<=h<=1):
                problems.append(f'[RANGE]  {split}:{name}.txt -> {x,y,w,h} not in [0,1]')
                ok_line = False
            if w <= 0 or h <= 0:
                problems.append(f'[SIZE]   {split}:{name}.txt -> non-positive w/h={w},{h}')
                ok_line = False
        if ok_line: n_ok += 1
    return n_ok, problems

def main():
    if len(sys.argv) < 2:
        print('Uso: python verify_dataset.py dataset/data.yaml'); sys.exit(1)
    cfg = load_yaml(sys.argv[1])
    base = cfg.get('path', '.'); nc = int(cfg['nc']); names = cfg['names']
    print(f'Base={base} | nc={nc} | names={names}')
    all_problems = []
    for split in ['train','valid','test']:
        split_dir = os.path.join(base, split)
        if not os.path.isdir(split_dir):
            print(f'Warn: split "{split}" no encontrado en {split_dir}')
            continue
        ok, probs = check_split(base, split, nc)
        print(f'{split}: OK={ok}, issues={len(probs)}')
        all_problems += probs
    if all_problems:
        print('\nIssues:'); [print(' -', p) for p in all_problems[:200]]
        if len(all_problems)>200: print(f'... (+{len(all_problems)-200})')
    else:
        print('\nTodo OK ✅')

if __name__ == '__main__':
    main()

# cd ".\prueba"
# python .\verify_dataset.py "..\dataset\data.yaml"
