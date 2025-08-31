import subprocess
import os

# Get the absolute path of the directory containing this script
script_dir = os.path.dirname(os.path.abspath(__file__))

# Command to run. "tiny-agents run" with "." means it will look for agent.json in the current directory.
command = ["tiny-agents", "run", "."]

print(f"Running agent from: {script_dir}")
print(f"Command: {' '.join(command)}")

# Run the command
# We use cwd to run the command from the script's directory
try:
    subprocess.run(command, cwd=script_dir, check=True)
except FileNotFoundError:
    print("Error: 'tiny-agents' command not found.")
    print("Please make sure you have installed huggingface_hub with mcp extras:")
    print("pip install -U 'huggingface_hub[mcp]'")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while running the agent: {e}") 