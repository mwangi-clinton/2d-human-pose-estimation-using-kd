import json

notebook_path = '/home/clinton-mwangi/Desktop/projects/2d-human-pose-estimation-using-kd/knowledge-distillation-for-2d-hpe.ipynb'

with open(notebook_path, 'r', encoding='utf-8') as f:
    nb = json.load(f)

# The first cell logic provided in the view_file output
# We want to insert `!pip install -r requirements.txt` after cloning and path setup
# or inside the colab block.

new_source = [
    "# @title Google colab\n",
    "%%capture\n",
    "import os\n",
    "import sys\n",
    "\n",
    "# Check if we are in Google Colab\n",
    "if 'google.colab' in str(get_ipython()):\n",
    "    print(\"Detected Google Colab. Cloning repository...\")\n",
    "    \n",
    "    REPO_URL = \"https://github.com/mwangi-clinton/2026-02-02-knowledge-distillation-for-2d-hpe.git\"\n",
    "    REPO_NAME = \"2026-02-02-knowledge-distillation-for-2d-hpe\"\n",
    "    if not os.path.exists(REPO_NAME):\n",
    "        !git clone $REPO_URL\n",
    "        # Move into the directory\n",
    "        %cd $REPO_NAME\n",
    "        # Install requirements\n",
    "        !pip install -r requirements.txt\n",
    "\n",
    "    sys.path.append(os.path.abspath(\".\"))"
]

nb['cells'][0]['source'] = new_source

with open(notebook_path, 'w', encoding='utf-8') as f:
    json.dump(nb, f, indent=1)
print("Notebook updated with pip install command in the first cell.")
