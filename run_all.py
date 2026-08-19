#!/usr/bin/env python3
"""Start llama-server, run the captioner, and stop only the server we started."""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
import tomllib
from pathlib import Path
from typing import Any
from urllib import error as urllib_error
from urllib import request as urllib_request


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_CONFIG = PROJECT_ROOT / "server_config.toml"


class ServerConfigError(RuntimeError):
    pass


def expanded_path(value: str, base: Path | None = None) -> Path:
    expanded = Path(os.path.expandvars(os.path.expanduser(value)))
    if not expanded.is_absolute() and base is not None:
        expanded = base / expanded
    return expanded.resolve()


def load_config(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ServerConfigError(
            f"Configuration not found: {path}\n"
            "Run setup first, then edit server_config.toml with your model paths."
        )
    try:
        with path.open("rb") as handle:
            data = tomllib.load(handle)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ServerConfigError(f"Could not read {path}: {exc}") from exc
    server = data.get("server")
    if not isinstance(server, dict):
        raise ServerConfigError(f"Missing [server] section in {path}")
    return server


def require_file(path: Path, label: str) -> None:
    if not path.is_file():
        raise ServerConfigError(f"{label} not found: {path}")


def build_server_command(config: dict[str, Any]) -> tuple[list[str], Path, str, int]:
    llama_cpp_dir = expanded_path(str(config.get("llama_cpp_dir", "")))
    executable_name = str(
        config.get(
            "executable", "llama-server.exe" if os.name == "nt" else "llama-server"
        )
    )
    executable = expanded_path(executable_name, llama_cpp_dir)
    model = expanded_path(str(config.get("model", "")))
    mmproj = expanded_path(str(config.get("mmproj", "")))
    mtp_value = str(config.get("mtp_assistant", "")).strip()
    mtp_assistant = expanded_path(mtp_value) if mtp_value else None
    template_value = str(config.get("chat_template_file", "")).strip()
    chat_template_file = (
        expanded_path(template_value, PROJECT_ROOT) if template_value else None
    )

    require_file(executable, "llama-server executable")
    require_file(model, "Main model")
    require_file(mmproj, "Multimodal projector")
    if mtp_assistant is not None:
        require_file(mtp_assistant, "MTP assistant model")
    if chat_template_file is not None:
        require_file(chat_template_file, "Chat template")

    host = str(config.get("host", "127.0.0.1"))
    port = int(config.get("port", 8080))
    media_path = expanded_path(
        str(config.get("media_path", PROJECT_ROOT / "input")), PROJECT_ROOT
    )
    media_path.mkdir(parents=True, exist_ok=True)

    command = [
        str(executable),
        "-m",
        str(model),
        "--mmproj",
        str(mmproj),
        "--media-path",
        str(media_path),
        "-c",
        str(int(config.get("context_size", 8192))),
        "-b",
        str(int(config.get("batch_size", 2048))),
        "-ub",
        str(int(config.get("ubatch_size", 2048))),
        "-ngl",
        str(config.get("gpu_layers", 999)),
        "-np",
        str(int(config.get("parallel_slots", 1))),
        "--host",
        host,
        "--port",
        str(port),
    ]

    if bool(config.get("fit", True)):
        command.extend(["-fit", "on"])
    if bool(config.get("flash_attention", True)):
        command.extend(["-fa", "on"])
    jinja_enabled = bool(config.get("jinja", True))
    if chat_template_file is not None and not jinja_enabled:
        raise ServerConfigError(
            "server.jinja must be true when server.chat_template_file is set"
        )
    if jinja_enabled:
        command.append("--jinja")
    if chat_template_file is not None:
        command.extend(["--chat-template-file", str(chat_template_file)])

    image_min_value = config.get("image_min_tokens")
    image_max_value = config.get("image_max_tokens")
    image_min_tokens = (
        int(image_min_value) if image_min_value not in (None, "") else None
    )
    image_max_tokens = (
        int(image_max_value) if image_max_value not in (None, "") else None
    )
    batch_size = int(config.get("batch_size", 2048))
    ubatch_size = int(config.get("ubatch_size", 2048))
    if batch_size < 1 or ubatch_size < 1:
        raise ServerConfigError(
            "server.batch_size and server.ubatch_size must be at least 1"
        )
    if ubatch_size > batch_size:
        raise ServerConfigError(
            "server.ubatch_size cannot exceed server.batch_size"
        )
    for label, value in (
        ("image_min_tokens", image_min_tokens),
        ("image_max_tokens", image_max_tokens),
    ):
        if value is not None and value < 1:
            raise ServerConfigError(f"server.{label} must be at least 1")
    if (
        image_min_tokens is not None
        and image_max_tokens is not None
        and image_min_tokens > image_max_tokens
    ):
        raise ServerConfigError(
            "server.image_min_tokens cannot exceed server.image_max_tokens"
        )
    if image_max_tokens is not None and ubatch_size < image_max_tokens:
        raise ServerConfigError(
            "server.ubatch_size must be at least server.image_max_tokens for "
            "Gemma 4 vision; otherwise llama.cpp can abort during non-causal "
            "image attention"
        )
    if image_min_tokens is not None:
        command.extend(["--image-min-tokens", str(image_min_tokens)])
    if image_max_tokens is not None:
        command.extend(["--image-max-tokens", str(image_max_tokens)])

    reasoning = str(config.get("reasoning", "")).strip()
    if reasoning:
        command.extend(["--reasoning", reasoning])

    split_mode = str(config.get("split_mode", "")).strip()
    if split_mode:
        command.extend(["--split-mode", split_mode])
    tensor_split = str(config.get("tensor_split", "")).strip()
    if tensor_split:
        command.extend(["--tensor-split", tensor_split])

    if mtp_assistant is not None:
        command.extend(
            [
                "--model-draft",
                str(mtp_assistant),
                "--spec-type",
                "draft-mtp",
                "--spec-draft-n-max",
                str(int(config.get("spec_draft_n_max", 4))),
            ]
        )
        p_min_value = config.get("spec_draft_p_min")
        if p_min_value not in (None, ""):
            command.extend(["--spec-draft-p-min", str(float(p_min_value))])

    extra_args = config.get("extra_args", [])
    if not isinstance(extra_args, list) or not all(
        isinstance(value, (str, int, float)) for value in extra_args
    ):
        raise ServerConfigError("server.extra_args must be a TOML array of strings/numbers")
    command.extend(str(value) for value in extra_args)
    return command, llama_cpp_dir, host, port


def health_url(host: str, port: int) -> str:
    return f"http://{host}:{port}/health"


def server_is_ready(url: str, timeout: float = 1.0) -> bool:
    try:
        with urllib_request.urlopen(url, timeout=timeout) as response:
            return response.status == 200
    except (urllib_error.URLError, TimeoutError, OSError):
        return False


def wait_for_server(
    process: subprocess.Popen[Any], url: str, startup_timeout: float
) -> None:
    deadline = time.monotonic() + startup_timeout
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise ServerConfigError(
                f"llama-server exited before becoming ready (exit code {return_code})"
            )
        if server_is_ready(url):
            return
        time.sleep(1)
    raise ServerConfigError(
        f"llama-server did not become ready within {startup_timeout:.0f} seconds"
    )


def stop_server(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    print("Stopping llama-server...")
    process.terminate()
    try:
        process.wait(timeout=15)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=5)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Start the configured local server and run batch_caption.py."
    )
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument(
        "--server-only",
        action="store_true",
        help="Start the server and keep it open without running the captioner.",
    )
    parser.add_argument(
        "--startup-timeout",
        type=float,
        default=None,
        help="Override server_config.toml startup_timeout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args, caption_args = parser.parse_known_args(argv)
    process: subprocess.Popen[Any] | None = None

    try:
        config_path = args.config.resolve()
        config = load_config(config_path)
        command, working_dir, host, port = build_server_command(config)
        ready_url = health_url(host, port)

        if server_is_ready(ready_url):
            raise ServerConfigError(
                f"A server is already responding at {ready_url}. "
                "Use run_captioner instead, or change the configured port."
            )

        print("Starting llama-server...")
        print(subprocess.list2cmdline(command))
        process = subprocess.Popen(command, cwd=working_dir)

        startup_timeout = (
            args.startup_timeout
            if args.startup_timeout is not None
            else float(config.get("startup_timeout", 180))
        )
        print(f"Waiting for {ready_url}...")
        wait_for_server(process, ready_url, startup_timeout)
        print("llama-server is ready.")

        if args.server_only:
            print("Server-only mode. Press Ctrl+C to stop.")
            return process.wait()

        endpoint = f"http://{host}:{port}/v1/chat/completions"
        caption_command = [
            sys.executable,
            str(PROJECT_ROOT / "batch_caption.py"),
            "--endpoint",
            endpoint,
            *caption_args,
        ]
        print("Running batch captioner...")
        return subprocess.run(caption_command, cwd=PROJECT_ROOT, check=False).returncode

    except ServerConfigError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("\nInterrupted.")
        return 130
    finally:
        if process is not None:
            stop_server(process)


if __name__ == "__main__":
    sys.exit(main())
