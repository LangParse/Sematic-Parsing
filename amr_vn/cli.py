
import argparse
from pathlib import Path
from typing import List

from .config import Paths, TrainConfig
from .data import AMRCorpusLoader, PublicTestLoader
from .preprocess import AMRPreprocessor
from .modeling import load_model
from .trainer import build_dataset, build_trainer
from .evaluator import bleu_on_samples

def cmd_build_json(args):
    paths = Paths(project_root=Path(args.project_root),
                  train_files=[Path(p) for p in args.train_files])
    samples = AMRCorpusLoader(paths.train_files).load()
    AMRCorpusLoader.save_json(samples, paths.json_path)
    print("Saved", paths.json_path)

def cmd_train(args):
    paths = Paths(project_root=Path(args.project_root))
    json_path = Path(args.json_path) if args.json_path else (paths.work_dir / "amr_data.json")
    samples = __load_json(json_path)
    cfg = TrainConfig(
        model_name=args.model_name,
        max_length=args.max_length,
        batch_size=args.batch_size,
        num_epochs=args.epochs,
        learning_rate=args.lr,
    )
    pre = AMRPreprocessor(cfg.model_name, cfg.max_length)
    from datasets import Dataset
    ds = Dataset.from_list(samples).map(pre)
    model = load_model(cfg.model_name)
    trainer = build_trainer(model, pre.tokenizer, ds, cfg, str(paths.model_dir))
    trainer.train()
    model.save_pretrained(str(paths.model_dir))
    pre.tokenizer.save_pretrained(str(paths.model_dir))
    print("Model saved to", paths.model_dir)

def cmd_eval(args):
    paths = Paths(project_root=Path(args.project_root))
    json_path = Path(args.json_path) if args.json_path else (paths.work_dir / "amr_data.json")
    samples = __load_json(json_path)
    model_path = args.model_path or str(paths.model_dir)
    from transformers import AutoTokenizer, T5ForConditionalGeneration
    tok = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    model = T5ForConditionalGeneration.from_pretrained(model_path)
    score = bleu_on_samples(model, tok, samples, limit=args.limit)
    print("BLEU:", score)

def cmd_predict(args):
    paths = Paths(project_root=Path(args.project_root),
                  test_file=Path(args.test_file))
    from .data import PublicTestLoader
    from .inference import AMRPredictor
    sents = PublicTestLoader(paths.test_file).load()
    predictor = AMRPredictor(args.model_path or str(paths.model_dir))
    out_path = paths.prediction_dir / "public_test_result.txt"
    predictor.predict_file(sents, out_path)
    print("Saved predictions to", out_path)

def __load_json(p: Path):
    import json
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def main():
    ap = argparse.ArgumentParser(prog="amr-vn")
    sub = ap.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build-json", help="Parse training AMR files into JSON")
    b.add_argument("--project-root", required=True)
    b.add_argument("--train-files", nargs="+", required=True, help="One or more AMR training files")
    b.set_defaults(func=cmd_build_json)

    t = sub.add_parser("train", help="Train model on JSON samples")
    t.add_argument("--project-root", required=True)
    t.add_argument("--json-path", default=None)
    t.add_argument("--model-name", default="VietAI/vit5-base")
    t.add_argument("--max-length", type=int, default=512)
    t.add_argument("--batch-size", type=int, default=4)
    t.add_argument("--epochs", type=int, default=10)
    t.add_argument("--lr", type=float, default=5e-5)
    t.set_defaults(func=cmd_train)

    e = sub.add_parser("eval", help="BLEU on a subset of JSON samples")
    e.add_argument("--project-root", required=True)
    e.add_argument("--json-path", default=None)
    e.add_argument("--model-path", default=None)
    e.add_argument("--limit", type=int, default=100)
    e.set_defaults(func=cmd_eval)

    p = sub.add_parser("predict", help="Predict AMR for public_test.txt")
    p.add_argument("--project-root", required=True)
    p.add_argument("--test-file", required=True)
    p.add_argument("--model-path", default=None)
    p.set_defaults(func=cmd_predict)

    args = ap.parse_args()
    args.func(args)

if __name__ == "__main__":
    main()
