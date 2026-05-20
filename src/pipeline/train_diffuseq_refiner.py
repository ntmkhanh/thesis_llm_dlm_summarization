import argparse
import os
import re
import subprocess
import sys
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="Fine-tune Shark-NLP/DiffuSeq for draft-summary refinement")
    p.add_argument("--diffuseq-dir", default="external/DiffuSeq")
    p.add_argument("--data-dir", required=True, help="Folder containing train.jsonl, valid.jsonl, test.jsonl")
    p.add_argument("--dataset", default="cnn_dailymail_refine")
    p.add_argument("--nproc-per-node", type=int, default=1)
    p.add_argument("--master-port", type=int, default=12233)
    p.add_argument("--diff-steps", type=int, default=2000)
    p.add_argument("--learning-steps", type=int, default=50000)
    p.add_argument("--save-interval", type=int, default=10000)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--microbatch", type=int, default=16)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--hidden-dim", type=int, default=128)
    p.add_argument("--hidden-t-dim", type=int, default=128)
    p.add_argument("--noise-schedule", default="sqrt")
    p.add_argument("--schedule-sampler", default="lossaware")
    p.add_argument("--vocab", default="bert")
    p.add_argument("--config-name", default="bert-base-uncased")
    p.add_argument("--use-plm-init", choices=["no", "bert"], default="no")
    p.add_argument("--seed", type=int, default=102)
    p.add_argument("--notes", default="cnn_dailymail_refine")
    return p.parse_args()


def patch_diffuseq_padding(diffuseq_dir: str):
    """Make DiffuSeq preprocessing less memory hungry on small machines."""
    path = Path(diffuseq_dir) / "diffuseq" / "text_datasets.py"
    if not path.exists():
        raise FileNotFoundError(f"DiffuSeq text dataset file not found: {path}")

    text = path.read_text(encoding="utf-8")
    patched = (
        "desc=f\"padding\",\n"
        "        batch_size=1,\n"
        "        load_from_cache_file=False,"
    )
    if patched in text:
        print(f"DiffuSeq padding patch already active in {path}")
        return

    old_blocks = [
        "        desc=f\"padding\",\n"
        "        batch_size=64,\n"
        "    )",
        "        desc=f\"padding\",\n"
        "    )",
    ]
    new = (
        "        desc=f\"padding\",\n"
        "        batch_size=1,\n"
        "        load_from_cache_file=False,\n"
        "    )"
    )
    for old in old_blocks:
        if old in text:
            path.write_text(text.replace(old, new), encoding="utf-8")
            print(f"Patched DiffuSeq padding batch_size=1 and disabled cache in {path}")
            return

    if "desc=f\"padding\"" in text:
        print(f"DiffuSeq padding map was found but not patched automatically in {path}")
        return
    print(f"DiffuSeq padding map not found in {path}")


def patch_diffuseq_numpy_aliases(diffuseq_dir: str):
    """Patch deprecated NumPy aliases used by the upstream DiffuSeq repo."""
    replacements = {
        "np.int": "int",
        "np.float": "float",
        "np.bool": "bool",
        "np.object": "object",
    }
    for path in Path(diffuseq_dir).rglob("*.py"):
        text = path.read_text(encoding="utf-8")
        new_text = text
        for old, new in replacements.items():
            new_text = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(old)}(?![A-Za-z0-9_])", new, new_text)
        if new_text != text:
            path.write_text(new_text, encoding="utf-8")
            print(f"Patched deprecated NumPy aliases in {path}")


def patch_diffuseq_wandb_stub(diffuseq_dir: str):
    """Disable wandb calls in the upstream DiffuSeq scripts."""
    path = Path(diffuseq_dir) / "wandb.py"
    stub = '''class _Config:
    def update(self, *args, **kwargs):
        return None


config = _Config()


def init(*args, **kwargs):
    return None


def log(*args, **kwargs):
    return None
'''
    if path.exists() and path.read_text(encoding="utf-8") == stub:
        return
    path.write_text(stub, encoding="utf-8")
    print(f"Installed DiffuSeq wandb no-op stub at {path}")


def main():
    args = parse_args()
    diffuseq_dir = os.path.abspath(args.diffuseq_dir)
    data_dir = os.path.abspath(args.data_dir)
    patch_diffuseq_padding(diffuseq_dir)
    patch_diffuseq_numpy_aliases(diffuseq_dir)
    patch_diffuseq_wandb_stub(diffuseq_dir)

    cmd = [
        sys.executable, "-m", "torch.distributed.launch",
        f"--nproc_per_node={args.nproc_per_node}",
        f"--master_port={args.master_port}",
        "--use_env",
        "scripts/run_train.py",
        "--diff_steps", str(args.diff_steps),
        "--lr", str(args.lr),
        "--learning_steps", str(args.learning_steps),
        "--save_interval", str(args.save_interval),
        "--seed", str(args.seed),
        "--noise_schedule", args.noise_schedule,
        "--hidden_dim", str(args.hidden_dim),
        "--hidden_t_dim", str(args.hidden_t_dim),
        "--bsz", str(args.batch_size),
        "--microbatch", str(args.microbatch),
        "--dataset", args.dataset,
        "--data_dir", data_dir,
        "--vocab", args.vocab,
        "--config_name", args.config_name,
        "--use_plm_init", args.use_plm_init,
        "--seq_len", str(args.seq_len),
        "--schedule_sampler", args.schedule_sampler,
        "--notes", args.notes,
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=diffuseq_dir, check=True)
    print(f"DiffuSeq checkpoints are under: {os.path.join(diffuseq_dir, 'diffusion_models')}")


if __name__ == "__main__":
    main()
