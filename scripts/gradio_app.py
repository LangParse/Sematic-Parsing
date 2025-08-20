import argparse
from pathlib import Path
from amr_vn.demo import GradioDemo


def main():
    ap = argparse.ArgumentParser()
    mx = ap.add_mutually_exclusive_group()
    mx.add_argument(
        "--model-path",
        default=str(Path(__file__).resolve().parents[1] / "outputs" / "models"),
    )
    mx.add_argument(
        "--hf-id",
        help="Hugging Face repo id, ví dụ: nphuoctho/semantic_parsing_amr_vit5",
    )
    ap.add_argument("--share", action="store_true")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--server-name", default="0.0.0.0")
    args, _ = ap.parse_known_args()  # tránh lỗi -f khi chạy trong notebook

    if args.hf_id:
        GradioDemo(hf_id=args.hf_id).launch(
            share=args.share, server_port=args.port, server_name=args.server_name
        )
    else:
        GradioDemo(model_path=args.model_path).launch(
            share=args.share, server_port=args.port, server_name=args.server_name
        )


if __name__ == "__main__":
    main()
