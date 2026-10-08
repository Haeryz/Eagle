"""Load W&B / Hugging Face credentials from local .env files into the environment (values are never printed).

Same convention as the user's other projects: lines like `wandb=<api key>` and `hf=<token>`. Files read, in order:
`Embodied/.env` (gitignored) and the shared .env at LOCANY_ENV_FILE (default: the Kaggle project's .env).
Existing environment variables are never overwritten. No secret is committed to this repo.
"""
import os
from pathlib import Path

WANDB_PROJECT = "LocateAnything"
REPO_ENV_FILE = Path(__file__).resolve().parents[2] / ".env"
SHARED_ENV_FILE = Path("/haeryz/banger/Kaggle/src/kaggle/.env")
KEY_TO_ENV = {"wandb": "WANDB_API_KEY", "hf": "HF_TOKEN"}


def load_env(*paths) -> None:
    paths = paths or (REPO_ENV_FILE, Path(os.environ.get("LOCANY_ENV_FILE", SHARED_ENV_FILE)))
    for path in map(Path, paths):
        if not path.exists():
            continue
        for line in path.read_text().splitlines():
            if "=" in line and not line.lstrip().startswith("#"):
                k, v = line.split("=", 1)
                env_name = KEY_TO_ENV.get(k.strip().lower())
                v = v.strip().strip('"').strip("'")
                if env_name and v:
                    os.environ.setdefault(env_name, v)
    os.environ.setdefault("WANDB_PROJECT", WANDB_PROJECT)
