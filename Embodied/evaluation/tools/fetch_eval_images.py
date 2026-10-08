"""Stream Rex-Omni-EvalData image tarballs and extract only the images listed in needed_images.txt,
so the multi-GB archives never have to be stored on disk."""
import argparse
import os
import tarfile

import requests
from huggingface_hub import hf_hub_url
from huggingface_hub.utils import build_hf_headers


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--needed", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--tars", nargs="+", default=["coco.tar.gz", "dense200.tar.gz", "sroie.tar.gz"])
    args = ap.parse_args()

    needed = {l.strip() for l in open(args.needed) if l.strip()}
    for tar_name in args.tars:
        url = hf_hub_url("Mountchicken/Rex-Omni-EvalData", tar_name, repo_type="dataset")
        with requests.get(url, headers=build_hf_headers(), stream=True, timeout=60) as r:
            r.raise_for_status()
            r.raw.decode_content = True
            n = 0
            with tarfile.open(fileobj=r.raw, mode="r|gz") as tf:
                for m in tf:
                    name = m.name.lstrip("./")
                    if m.isfile() and name in needed:
                        dst = os.path.join(args.out_dir, name)
                        os.makedirs(os.path.dirname(dst), exist_ok=True)
                        with open(dst, "wb") as f:
                            f.write(tf.extractfile(m).read())
                        n += 1
        print(f"{tar_name}: extracted {n}", flush=True)


if __name__ == "__main__":
    main()
