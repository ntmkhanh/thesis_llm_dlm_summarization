import argparse
import os
import subprocess
import sys


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
    p.add_argument("--seed", type=int, default=102)
    p.add_argument("--notes", default="cnn_dailymail_refine")
    return p.parse_args()


def main():
    args = parse_args()
    diffuseq_dir = os.path.abspath(args.diffuseq_dir)
    data_dir = os.path.abspath(args.data_dir)

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
        "--seq_len", str(args.seq_len),
        "--schedule_sampler", args.schedule_sampler,
        "--notes", args.notes,
    ]
    print("$", " ".join(cmd))
    subprocess.run(cmd, cwd=diffuseq_dir, check=True)
    print(f"DiffuSeq checkpoints are under: {os.path.join(diffuseq_dir, 'diffusion_models')}")


if __name__ == "__main__":
    main()
