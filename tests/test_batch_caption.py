from __future__ import annotations

import base64
import sys
import tempfile
import unittest
from pathlib import Path

from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import batch_caption  # noqa: E402
import run_all  # noqa: E402


class BatchCaptionTests(unittest.TestCase):
    def test_recent_batch_cli_aliases(self) -> None:
        args = batch_caption.build_parser().parse_args(
            ["--base-url", "http://127.0.0.1:9000/v1", "--limit", "3"]
        )
        self.assertEqual(args.endpoint, "http://127.0.0.1:9000/v1")
        self.assertEqual(args.max_images, 3)

    def test_normalize_endpoint_accepts_v1_base(self) -> None:
        self.assertEqual(
            batch_caption.normalize_endpoint("http://127.0.0.1:8080/v1/"),
            "http://127.0.0.1:8080/v1/chat/completions",
        )

    def test_discover_images_and_mirror_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "input"
            output_dir = root / "output"
            nested = input_dir / "portraits"
            nested.mkdir(parents=True)
            (input_dir / "one.JPG").write_bytes(b"x")
            (nested / "two.tiff").write_bytes(b"x")
            (nested / "ignore.txt").write_text("x", encoding="utf-8")

            images = batch_caption.discover_images(input_dir, recursive=True)

            self.assertEqual(len(images), 2)
            self.assertEqual(
                batch_caption.output_path_for(nested / "two.tiff", input_dir, output_dir),
                output_dir / "portraits" / "two.txt",
            )

    def test_nonrecursive_discovery_ignores_subfolders(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            input_dir = Path(temp)
            (input_dir / "nested").mkdir()
            (input_dir / "top.png").write_bytes(b"x")
            (input_dir / "nested" / "child.png").write_bytes(b"x")
            images = batch_caption.discover_images(input_dir, recursive=False)
            self.assertEqual(images, [input_dir / "top.png"])

    def test_output_collision_is_detected(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            input_dir = root / "input"
            output_dir = root / "output"
            input_dir.mkdir()
            jpg = input_dir / "same.jpg"
            png = input_dir / "same.png"
            jpg.write_bytes(b"x")
            png.write_bytes(b"x")
            collisions = batch_caption.find_output_collisions(
                [jpg, png], input_dir, output_dir
            )
            self.assertEqual(collisions[output_dir / "same.txt"], [jpg, png])

    def test_extract_caption_prefers_content_and_strips_thinking(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "<think>internal notes</think> A red fox in snow."
                    },
                    "finish_reason": "stop",
                }
            ]
        }
        result = batch_caption.extract_caption(response)
        self.assertEqual(result.caption, "A red fox in snow.")
        self.assertEqual(result.response_field, "content")
        self.assertEqual(result.finish_reason, "stop")

    def test_extract_caption_falls_back_to_reasoning_content(self) -> None:
        response = {
            "choices": [
                {
                    "message": {
                        "content": "",
                        "reasoning_content": [{"type": "text", "text": "A blue vase."}],
                    }
                }
            ]
        }
        result = batch_caption.extract_caption(response)
        self.assertEqual(result.caption, "A blue vase.")
        self.assertEqual(result.response_field, "reasoning_content")

    def test_extract_caption_rejects_empty_response(self) -> None:
        with self.assertRaises(batch_caption.CaptionError):
            batch_caption.extract_caption({"choices": [{"message": {"content": ""}}]})

    def test_bmp_is_normalized_to_a_supported_data_url(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            image_path = Path(temp) / "sample.bmp"
            Image.new("RGB", (8, 6), (20, 40, 60)).save(image_path)
            url = batch_caption.image_as_data_url(image_path)
            header, encoded = url.split(",", 1)
            self.assertEqual(header, "data:image/jpeg;base64")
            self.assertGreater(len(base64.b64decode(encoded)), 10)

    def test_relative_file_url_encodes_spaces(self) -> None:
        input_dir = Path("C:/caption/input")
        image_path = input_dir / "nested folder" / "my image.png"
        self.assertEqual(
            batch_caption.image_as_file_url(image_path, input_dir),
            "file://nested%20folder/my%20image.png",
        )

    def test_server_command_includes_multimodal_and_mtp_models(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            llama_dir = root / "llama"
            llama_dir.mkdir()
            executable = llama_dir / "llama-server"
            model = root / "model.gguf"
            mmproj = root / "mmproj.gguf"
            assistant = root / "assistant.gguf"
            chat_template = root / "chat_template.jinja"
            for path in (executable, model, mmproj, assistant, chat_template):
                path.write_bytes(b"x")

            command, working_dir, host, port = run_all.build_server_command(
                {
                    "llama_cpp_dir": str(llama_dir),
                    "executable": "llama-server",
                    "model": str(model),
                    "mmproj": str(mmproj),
                    "mtp_assistant": str(assistant),
                    "chat_template_file": str(chat_template),
                    "media_path": str(root / "input"),
                    "host": "127.0.0.1",
                    "port": 8080,
                    "tensor_split": "1,1",
                    "batch_size": 2048,
                    "ubatch_size": 2048,
                    "spec_draft_n_max": 4,
                    "image_min_tokens": 70,
                    "image_max_tokens": 1120,
                }
            )

            self.assertEqual(working_dir, llama_dir)
            self.assertEqual((host, port), ("127.0.0.1", 8080))
            self.assertIn("--mmproj", command)
            self.assertIn("--model-draft", command)
            self.assertIn("draft-mtp", command)
            self.assertIn("1,1", command)
            self.assertIn("4", command)
            self.assertIn("--chat-template-file", command)
            self.assertIn("--image-min-tokens", command)
            self.assertIn("--image-max-tokens", command)
            self.assertIn("1120", command)
            self.assertIn("-ub", command)
            self.assertIn("2048", command)
            self.assertNotIn("--spec-draft-p-min", command)

    def test_server_command_rejects_inverted_image_token_range(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            llama_dir = root / "llama"
            llama_dir.mkdir()
            executable = llama_dir / "llama-server"
            model = root / "model.gguf"
            mmproj = root / "mmproj.gguf"
            for path in (executable, model, mmproj):
                path.write_bytes(b"x")

            with self.assertRaises(run_all.ServerConfigError):
                run_all.build_server_command(
                    {
                        "llama_cpp_dir": str(llama_dir),
                        "executable": "llama-server",
                        "model": str(model),
                        "mmproj": str(mmproj),
                        "image_min_tokens": 1120,
                        "image_max_tokens": 70,
                    }
                )

    def test_server_command_rejects_ubatch_smaller_than_vision_budget(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            llama_dir = root / "llama"
            llama_dir.mkdir()
            executable = llama_dir / "llama-server"
            model = root / "model.gguf"
            mmproj = root / "mmproj.gguf"
            for path in (executable, model, mmproj):
                path.write_bytes(b"x")

            with self.assertRaisesRegex(
                run_all.ServerConfigError, "ubatch_size must be at least"
            ):
                run_all.build_server_command(
                    {
                        "llama_cpp_dir": str(llama_dir),
                        "executable": "llama-server",
                        "model": str(model),
                        "mmproj": str(mmproj),
                        "batch_size": 2048,
                        "ubatch_size": 512,
                        "image_max_tokens": 1120,
                    }
                )


if __name__ == "__main__":
    unittest.main()
