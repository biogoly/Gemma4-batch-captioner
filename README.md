<img width="2172" height="724" alt="image" src="https://github.com/user-attachments/assets/92c79764-aa19-496a-bc48-5ab9b55e19d5" />


# Gemma Batch Captioner

Turn a folder of images into detailed natural-language reconstruction prompts using a local multimodal model served by `llama-server`.

The default configuration uses the **standard Gemma 4 12B Instruct QAT Q4_0 release**. The higher-quality **Gemma 4 31B Instruct QAT Q4_0 release** is also supported. Both profiles use a BF16 multimodal projector and matching Q4_0 MTP assistant; model files are not included.

## What it does

- Processes images **one at a time**, so folder size does not cause the whole batch to be loaded into memory.
- Reads from `input/` and writes same-named `.txt` captions under `output/`.
- Recursively scans subfolders and mirrors their layout in `output/`.
- Supports `.jpg`, `.jpeg`, `.png`, `.webp`, `.bmp`, `.gif`, `.tif`, and `.tiff`.
- Uses an editable prompt at `prompts/caption_prompt.txt`.
- Skips non-empty existing captions by default, with an overwrite option.
- Retries failed requests and writes a crash-resistant CSV audit log after every image.
- Handles normal `content`, list-style content, and `reasoning_content` responses.
- Removes common `<think>`, `<analysis>`, and `<reasoning>` blocks from saved captions.
- Can start `llama-server`, wait until it is ready, run the batch, and stop only the server process it launched.

The default prompt is aimed at prompt reconstruction for modern natural-language image models such as Flux, rather than short alt text or old Stable Diffusion tag soup.

## Recommended model profiles

The recommended files are Unsloth's standard instruction-tuned QAT releases. Their QAT Q4_0 target GGUF is labeled `UD-Q4_K_XL` in the repository.

| Profile | Main GGUF | Projector | MTP assistant | Best use |
| --- | --- | --- | --- | --- |
| Gemma 4 12B QAT Q4_0 | [`gemma-4-12B-it-qat-UD-Q4_K_XL.gguf`](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF/blob/main/gemma-4-12B-it-qat-UD-Q4_K_XL.gguf) | [`mmproj-BF16.gguf`](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF/blob/main/mmproj-BF16.gguf) | [`mtp-gemma-4-12B-it.gguf`](https://huggingface.co/unsloth/gemma-4-12B-it-qat-GGUF/blob/main/mtp-gemma-4-12B-it.gguf) | Fast batches and lower memory use; project default. |
| Gemma 4 31B QAT Q4_0 | [`gemma-4-31B-it-qat-UD-Q4_K_XL.gguf`](https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF/blob/main/gemma-4-31B-it-qat-UD-Q4_K_XL.gguf) | [`mmproj-BF16.gguf`](https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF/blob/main/mmproj-BF16.gguf) | [`mtp-gemma-4-31B-it.gguf`](https://huggingface.co/unsloth/gemma-4-31B-it-qat-GGUF/blob/main/mtp-gemma-4-31B-it.gguf) | Quality-first descriptive captioning; slower and heavier, but generally more detailed and precise. |

Keep all three files in a profile at the same model size. Do not pair a 12B main model with the 31B projector or MTP assistant, or vice versa.

To use the 31B profile, replace the three model paths in `server_config.toml`:

```toml
model = 'C:\Users\YOUR_NAME\LLM\models\Gemma-4-31B-QAT\gemma-4-31B-it-qat-UD-Q4_K_XL.gguf'
mmproj = 'C:\Users\YOUR_NAME\LLM\models\Gemma-4-31B-QAT\mmproj-BF16.gguf'
mtp_assistant = 'C:\Users\YOUR_NAME\LLM\models\Gemma-4-31B-QAT\mtp-gemma-4-31B-it.gguf'
```

### Optional Heretic/abliterated models

Heretic or abliterated Gemma 4 checkpoints can be substituted when uncensored or NSFW descriptions are specifically desired. Use a QAT vision checkpoint based on the same Gemma 4 architecture and keep its projector and MTP assistant matched to the same 12B or 31B size.

## Folder layout

```text
gemma-batch-captioner/
├── batch_caption.py
├── run_all.py
├── setup.bat
├── run_all.bat
├── run_captioner.bat
├── start_server.bat
├── server_config.example.toml
├── input/
├── output/
├── prompts/
│   └── caption_prompt.txt
├── templates/
│   └── gemma4_chat_template.jinja
├── logs/
└── tests/
```

## Windows quick start

1. Install Python 3.11 or newer and a current CUDA build of `llama.cpp` with Gemma 4 multimodal and MTP support.
2. Download all three files from either recommended model profile above. Start with 12B for speed or use 31B when descriptive quality matters more.
3. Double-click `setup.bat`.
4. Open `server_config.toml`, which setup creates from the example, and replace the three model paths plus `llama_cpp_dir`.
5. Put images in `input/`.
6. Edit `prompts/caption_prompt.txt` if desired.
7. Double-click `run_all.bat`.

Captions appear in `output/`; the CSV audit trail appears in `logs/`.

The example contains the exact standard 12B filenames; only their parent folders are placeholders.

### Dual GPUs

For two matching GPUs, set:

```toml
split_mode = 'layer'
tensor_split = '1,1'
```

For one GPU, leave `tensor_split = ''`.

## Linux quick start

```bash
chmod +x setup.sh run_all.sh run_captioner.sh
./setup.sh
```

Edit `server_config.toml`, set `executable = 'llama-server'`, add images, and run:

```bash
./run_all.sh
```

## Two ways to run

### One-click managed server

`run_all.bat` or `./run_all.sh` starts the configured server, polls `/health`, captions the batch, and shuts down that server in a `finally` block.

To keep only the server running:

```text
start_server.bat
```

### Use a server already running

Run `run_captioner.bat` or:

```bash
python batch_caption.py --endpoint http://127.0.0.1:8080/v1/chat/completions
```

The endpoint may also end at `/v1`; the script adds `/chat/completions` automatically. `--base-url` is an alias for `--endpoint`.

## Useful options

```bash
# Preview source-to-output mappings without calling the model
python batch_caption.py --dry-run

# Test only ten images (--limit is an alias)
python batch_caption.py --limit 10 --run-label gemma-test

# Replace existing non-empty captions
python batch_caption.py --overwrite

# Scan only the top level of input/
python batch_caption.py --no-recursive

# Reduce very large images before sending them
python batch_caption.py --max-image-megapixels 12

# Use a shorter output budget
python batch_caption.py --max-tokens 1200
```

Run `python batch_caption.py --help` for every option.

## Image transport

The default is `--image-transport data-url`. Each image is base64-encoded into its request. This is portable and avoids Windows `file://` and `--media-path` headaches.

For less client-side encoding, use:

```bash
python batch_caption.py --image-transport file-url
```

That mode sends paths relative to `input/`. The server must be launched with `--media-path` pointing to that exact input folder. The managed launcher already supplies it.

TIFF, BMP, and GIF inputs are normalized with Pillow before data-URL submission. GIF captioning uses the first frame.

## Output and logs

For this input:

```text
input/portraits/example.tiff
```

the caption is written to:

```text
output/portraits/example.txt
```

If two files in the same folder share a stem, such as `example.jpg` and `example.png`, the program stops before processing because both would map to `example.txt`.

Each run creates `logs/caption_run_YYYYMMDD_HHMMSS.csv` with source and output paths, timing, attempts, response field, finish reason, status, errors, and raw server response. Caption text is never echoed to the terminal; only progress and errors are shown.

## Gemma and MTP configuration

### Current Gemma 4 chat template

The project includes Google's canonical Gemma 4 template published on July 9, 2026, pinned from [`google/gemma-4-12B-it`](https://huggingface.co/google/gemma-4-12B-it/blob/711c1368e39f1712f48ff0eb7bcdbbb760d52db0/chat_template.jinja). The example config loads it separately from the older template embedded in the GGUF:

```text
--jinja
--chat-template-file templates/gemma4_chat_template.jinja
```

Keep `--jinja` enabled and before the template-file option; current `llama.cpp` requires it for arbitrary external templates. Set `chat_template_file = ''` to fall back to the template embedded in the GGUF.

To refresh the bundled template from Google's current model repository:

```powershell
curl.exe -L "https://huggingface.co/google/gemma-4-12B-it/raw/main/chat_template.jinja" -o "templates\gemma4_chat_template.jinja"
```

An upstream refresh can change prompt formatting, so test a few known images before replacing the pinned copy.

### Vision token budget

Gemma 4 supports per-image budgets of 70, 140, 280, 560, or 1120 visual tokens. The example uses a dynamic range of 70–1120:

```text
--image-min-tokens 70
--image-max-tokens 1120
```

The 1120 maximum preserves more fine detail than the model's usual 280-token setting, which suits reconstruction captions. It also costs more context, processing time, and memory. If you hit memory pressure or want faster batches, lower `image_max_tokens` to 560 or 280. For small or simple images, the dynamic minimum still lets the server use fewer tokens.

Gemma 4's image pass uses non-causal attention, so the physical micro-batch must be large enough to hold the selected image budget. The example therefore also sets:

```text
--batch-size 2048
--ubatch-size 2048
```

Keep `ubatch_size >= image_max_tokens` and `batch_size >= ubatch_size`. The managed launcher validates both relationships before starting the server.

### MTP assistant

When `mtp_assistant` is non-empty, the managed launcher adds:

```text
--model-draft <assistant.gguf>
--spec-type draft-mtp
--spec-draft-n-max 4
```

No `--spec-draft-p-min` override is emitted unless you explicitly add `spec_draft_p_min` to `server_config.toml`; the example leaves llama.cpp's default alone.

The example also enables `--reasoning off`, because this tool wants the final reconstruction prompt rather than visible deliberation. The Python parser still tolerates servers that return useful text through `reasoning_content`.

If MTP plus flash attention crashes on a particular build or GPU, first update `llama.cpp`, then try `flash_attention = false` and/or reduce `image_max_tokens`. Also verify that the main, projector, and assistant files all match the selected Gemma 4 12B or 31B QAT profile.

### `non-causal attention requires n_ubatch >= n_tokens`

This assertion means the image token budget is larger than llama.cpp's physical micro-batch. The stock micro-batch default is 512, which is too small for this project's 1120-token vision maximum. Set the following in `server_config.toml`, restart the server, and rerun the batch:

```toml
batch_size = 2048
ubatch_size = 2048
image_max_tokens = 1120
```

If 2048 causes memory pressure, use `ubatch_size = 1024` with `image_max_tokens = 560`. Do not keep the 1120-token maximum with a smaller micro-batch.

## Tests

```bash
python -m unittest discover -s tests -v
```

GitHub Actions runs the same suite on Python 3.11 and 3.12.

## License

MIT. Model weights remain subject to their own licenses and are not distributed with this project.
