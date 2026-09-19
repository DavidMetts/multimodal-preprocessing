"""Run locally: python package_folder.py path/to/events collection.zip"""
import argparse
from pathlib import Path
from preprocessing.inputs import folder_zip

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('folder'); p.add_argument('output')
    args=p.parse_args(); data=folder_zip(args.folder)
    with Path(args.output).open('xb') as f: f.write(data)
    print('Created',args.output,'— upload this ZIP to the dashboard.')
