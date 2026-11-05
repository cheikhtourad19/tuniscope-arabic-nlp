"""Read-only progress of an encoder run, including before results.json exists."""
import argparse
import json
from pathlib import Path


def progress(directory,expected_models=12):
    complete=[];pending=[]
    for folder in sorted(directory.iterdir()):
        if not folder.is_dir() or folder.name=='figures':continue
        if (folder/'tuniscope.json').is_file():complete.append(folder.name)
        elif (folder/'history.json').is_file():
            try:history=json.loads((folder/'history.json').read_text())
            except json.JSONDecodeError:continue # A writer may be updating this snapshot.
            pending.append((folder.name,history[-1]['epoch'] if history else 0))
        else:pending.append((folder.name,0))
    print(f'Apprentissages terminés : {len(complete)}/{expected_models}')
    for name,epoch in pending:print(f'En cours : {name} — dernière époque enregistrée : {epoch}')
    result=directory/'results.json'
    if result.exists():
        print('Résultats complets présents ; contrôle de rechargement :', 'présent' if (directory/'verification.json').exists() else 'encore à effectuer')
    else:print('Pas encore de résultats complets : ne pas présenter les fichiers partiels comme un run final.')


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('run',type=Path);p.add_argument('--expected-models',type=int,default=12)
    a=p.parse_args();progress(a.run,a.expected_models)
