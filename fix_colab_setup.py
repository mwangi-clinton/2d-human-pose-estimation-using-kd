import json

notebook_path = '/home/clinton-mwangi/Desktop/projects/2d-human-pose-estimation-using-kd/knowledge-distillation-for-2d-hpe.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# Improved robust robust startup logic
# 1. Force move to /content (default colab dir)
# 2. Check for repo
# 3. Clone or pull
# 4. Install requirements
# 5. Add to path (safely)

new_source = [
    "# @title Google colab\n",
    "%%capture\n",
    "import os\n",
    "import sys\n",
    "import shutil\n",
    "\n",
    "# Check if we are in Google Colab\n",
    "if 'google.colab' in str(get_ipython()):\n",
    "    print(\"Detected Google Colab.\")\n",
    "    \n",
    "    # 1. Reset to standard startup directory to avoid FileNotFoundError on getcwd()\n",
    "    #    if the previous directory was deleted.\n",
    "    try:\n",
    "        os.chdir('/content')\n",
    "    except OSError:\n",
    "        pass # Already there or can't move, proceed carefully\n",
    "        \n",
    "    REPO_URL = \"https://github.com/mwangi-clinton/2026-02-02-knowledge-distillation-for-2d-hpe.git\"\n",
    "    REPO_NAME = \"2026-02-02-knowledge-distillation-for-2d-hpe\"\n",
    "    \n",
    "    # 2. Clone if not exists, otherwise enter\n",
    "    if not os.path.exists(REPO_NAME):\n",
    "        print(\"Cloning repository...\")\n",
    "        !git clone $REPO_URL\n",
    "        %cd $REPO_NAME\n",
    "        # Install dependencies only on first clone/setup\n",
    "        !pip install -r .requirements_colab.txt\n",
    "    else:\n",
    "        print(\"Repository exists.\")\n",
    "        %cd $REPO_NAME\n",
    "        # Optional: !git pull origin main\n",
    "\n",
    "    # 3. Safely add to path\n",
    "    cwd = os.getcwd()\n",
    "    if cwd not in sys.path:\n",
    "        sys.path.append(cwd)\n",
    "    print(f\"Current working directory: {cwd}\")"
]

nb['cells'][0]['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook updated with robust Colab setup cell.")
