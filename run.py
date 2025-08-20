from pathlib import Path

from amr_vn.cli import cmd_build_json, cmd_train, cmd_eval, cmd_predict
from amr_vn.demo import GradioDemo


def ask_path(prompt, default=None):
    p = input(f"{prompt} [{default}]: ").strip()
    return p or (default or "")


def main():
    project_root = Path(__file__).parent.resolve()
    print("=== AMR Vietnamese Runner ===")
    print(f"Project root: {project_root}")
    while True:
        print("\nChọn tác vụ:")
        print(" 1) Build JSON từ nhiều file train")
        print(" 2) Train")
        print(" 3) Eval nhanh (BLEU)")
        print(" 4) Predict trên public_test")
        print(" 5) Demo Gradio")
        print(" 6) Thoát")
        choice = input("> ").strip()

        if choice == "1":
            # Build JSON
            print("\n-- Build JSON --")
            default_train = str(project_root / "data" / "train" / "train_split.txt")
            train_files = input(
                f"Đường dẫn các file train, cách nhau bởi dấu cách \n   (vd: {default_train})\n> "
            ).strip()
            if not train_files:
                train_files = default_train
            args = type("Args", (), {})()
            args.project_root = str(project_root)
            args.train_files = train_files.split()
            cmd_build_json(args)

        elif choice == "2":
            # Train
            print("\n-- Train --")
            default_json = str(project_root / "outputs" / "amr_data.json")
            default_model = "VietAI/vit5-base"
            args = type("Args", (), {})()
            args.project_root = str(project_root)
            args.json_path = ask_path("JSON samples", default_json)
            args.model_name = ask_path("Model name", default_model)
            args.max_length = int(ask_path("Max length", "512"))
            args.batch_size = int(ask_path("Batch size", "4"))
            args.epochs = int(ask_path("Epochs", "10"))
            args.lr = float(ask_path("Learning rate", "5e-5"))
            cmd_train(args)

        elif choice == "3":
            # Eval
            print("\n-- Eval --")
            default_json = str(project_root / "outputs" / "amr_data.json")
            default_model_path = str(project_root / "outputs" / "models")
            args = type("Args", (), {})()
            args.project_root = str(project_root)
            args.json_path = ask_path("JSON samples", default_json)
            args.model_path = ask_path("Model path", default_model_path)
            args.limit = int(ask_path("Số mẫu đánh giá", "100"))
            cmd_eval(args)

        elif choice == "4":
            # Predict
            print("\n-- Predict --")
            default_test = str(project_root / "data" / "test" / "public_test.txt")
            default_model_path = str(project_root / "outputs" / "models")
            args = type("Args", (), {})()
            args.project_root = str(project_root)
            args.test_file = ask_path("File test", default_test)
            args.model_path = ask_path("Model path", default_model_path)
            cmd_predict(args)

        elif choice == "5":
            print("\n-- Demo Gradio --")
            print("Chọn nguồn model: 1) Local  2) HuggingFace")
            src = input("> ").strip() or "2"
            share = input("Share public? [y/N]: ").strip().lower() == "y"
            port = int(ask_path("Server port", "7860"))
            if src == "1":
                default_model_path = str(project_root / "outputs" / "models")
                mp = ask_path("Model path", default_model_path)
                GradioDemo(model_path=mp).launch(share=share, server_port=port)
            else:
                hf = ask_path("HF repo id", "nphuoctho/semantic_parsing_amr_vit5")
                GradioDemo(hf_id=hf).launch(share=share, server_port=port)

        elif choice == "6":
            print("Thoát.")
            return
        else:
            print("Lựa chọn không hợp lệ.")


if __name__ == "__main__":
    main()
