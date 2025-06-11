import subprocess
import multiprocessing
import os
import time
import argparse
import sys

def check_node_installed():
    try:
        # Run the 'node -v' command to check Node.js version
        result = subprocess.run(['node', '-v'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        if result.returncode == 0:
            print(f"Node.js is installed. Version: {result.stdout.strip()}")
            return 1
        else:
            print("Node.js is not installed.")
            return 0
    except FileNotFoundError:
        print("Node.js is not installed.")
        return 0

# setup the chrome environment for vm
def setup():
    if not check_node_installed():
        subprocess.run('curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.39.3/install.sh | bash', shell=True)
        subprocess.run('source ~/.bashrc', shell=True)
        subprocess.run('nvm install 18.16', shell=True)

    if not os.path.exists('./chrome-linux'):
        subprocess.run("wget -q 'https://www.googleapis.com/download/storage/v1/b/chromium-browser-snapshots/o/Linux_x64%2F978038%2Fchrome-linux.zip?generation=1646544045015587&alt=media' -O ./chrome_97.zip && unzip ./chrome_97.zip -d ./", shell=True)
        
        subprocess.run('npm i', shell=True)
        # os.system('sudo apt install npm@9.6.5')

# copy the session
def copy_session():
    if os.path.exists('./saved_session'):
        subprocess.run('cp -r saved_session temp_session', shell=True)
    else:
        print('NO SAVED SESSION FOUND! Create an aunthenticated session and save it in ./saved_session')
        sys.exit(0)

# Function to create a directory with 777 permission
def create_directory(dir_name):
    if not os.path.exists(dir_name):
        os.makedirs(dir_name)
        os.chmod(dir_name, 0o777)
        print(f"Created {dir_name} with 777 permissions.")
    else:
        print(f"{dir_name} already exists.")
        print('CLEANING THE DIRECTORY OF ANY FILES')
        subprocess.run(f'rm -rf ./{dir_name}/*', shell=True)

def create_data_directories():
    # Directories to be created
    dir1 = "./control"
    dir2 = "./adblock"

    # Create both directories
    create_directory(dir1)
    create_directory(dir2)

def check_and_start_container(container_name, image_name, extn):
    """Check if a container is running and start it if it's not."""
    result = subprocess.run(["docker", "ps", "-q", "-f", f"name={container_name}"], capture_output=True, text=True)
    if not result.stdout.strip():
        print(f"Starting container: {container_name}")
        subprocess.run(["docker", "run", "-d", "--name", container_name, "-v", f"./{extn}:/{extn}", image_name, 'tail', '-f', '/dev/null'])
    else:
        print(f"Container {container_name} is already running.")

def feed_url_to_container(container_name, url, extn):
    command = f'docker exec -i {container_name} python3 docker_in.py --url {url} --extn {extn}'
    subprocess.run(command, shell=True)

def handle_container(container_name, image_name, url, extn):
    check_and_start_container(container_name, image_name, extn)
    feed_url_to_container(container_name, url, extn)
    time.sleep(1)  # Add delay if necessary

# List of URLs to be crawled
urls = open('websites_inner_sites.txt', 'r').read().splitlines()
parser = argparse.ArgumentParser(description='Specify Extension (control/adblock) and Auth mode (0/1) for wrapper_out.py')
parser.add_argument('--extn', type=str, default='control')
parser.add_argument('--auth', type=int, default=0)
args = parser.parse_args()

create_data_directories()

if args.auth == 0:
    # Docker container names
    containers = ["accads_control", "accads_adblock"]

    # Build Docker images (assuming Dockerfiles are in the current directory)
    subprocess.run(["docker", "build", "-t", "accads", "-f", "Dockerfile", "."])
    for url in urls:
        # Create multiprocessing processes
        processes = []
        # image_name = "accads"
        image_name = "racro/accads:1.0.0"
        p1 = multiprocessing.Process(target=handle_container, args=(containers[0], image_name, url, 'control'))
        p2 = multiprocessing.Process(target=handle_container, args=(containers[1], image_name, url, 'adblock'))
        
        p1.start()
        p2.start()

        TIMEOUT = 90
        start = time.time()
        print("joining jobs")
        # Wait for all processes to finish
        p1.join(timeout = 60)
        p2.join(timeout = 60)

        while time.time() - start <= TIMEOUT:
            if p1.is_alive():
                p1.terminate()
            if p2.is_alive():
                p2.terminate()

        time.sleep(2)

elif args.auth == 1:
    setup()
    for url in urls:
        try:
            tries = 3
            while tries > 0:
                try:
                    # Execute a command with a timeout of 5 seconds
                    result = subprocess.run(['python3', 'wrapper_in.py', '--url', url, '--extn', args.extn], stdout = subprocess.PIPE, stderr = subprocess.PIPE)
                    print("Command completed:", result)

                    break
                except Exception as e:
                    print(e)
                    tries -= 1

        except subprocess.TimeoutExpired:
            print("Command timed out and was terminated.")

        time.sleep(2)

else:
    print('PLEASE SPECIFY Auth to be either 0 (Unauthenticated) or 1 (Authenticated)')
