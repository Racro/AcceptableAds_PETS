import argparse
import os
import time
import subprocess

parser = argparse.ArgumentParser(description='Specify URL and EXTN for wrapper_in.py')
parser.add_argument('--url', type=str)
parser.add_argument('--extn', type=str)
args = parser.parse_args()

os.chdir('./accads_crawler')

# cmd = f'npm run crawl -- -u {args.url} -o /{args.extn}/ -v -f -d "requests,cookies,ads,screenshots,cmps,videos" --reporters "cli,file" -l /{args.extn}/ --autoconsent-action "optIn" --extn {args.extn}'
cmd = [
    'npm', 'run', 'crawl', '--',
    '-u', args.url,
    '-o', f'/{args.extn}/',
    '-v', '-f', '-d', 'requests,cookies,ads,screenshots,cmps,videos',
    '--reporters', 'cli,file',
    '-l', f'/{args.extn}/',
    '--autoconsent-action', 'optIn',
    '--extn', args.extn
]

try:
    process = subprocess.run(cmd, check=True, text=True, capture_output=True)
    print(process.stdout)
except subprocess.CalledProcessError as e:
    print(f"Error executing command: {e}")
    print(f"Error output: {e.stderr}")
    raise