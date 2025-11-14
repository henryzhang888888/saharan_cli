import os
import io
import base64
import argparse
import time
import uuid

from PIL import Image
from openai import OpenAI

# Target Amazon A+ spec
TARGET_W, TARGET_H = 970, 600
# Closest valid OpenAI generation size
GEN_W, GEN_H = 1536, 1024

SYSTEM_MESSAGE = (
    "Create a premium 970x600 Amazon A+ hero module that is typography-first. "
    "Use the provided text exactly, keep the layout modern, and emphasize clarity."
)

def save_b64_to_file(b64: str, path: str):
    data = base64.b64decode(b64)
    with open(path, "wb") as f:
        f.write(data)


def image_to_bytes(img: Image.Image, format: str = "PNG") -> bytes:
    buf = io.BytesIO()
    img.save(buf, format=format)
    return buf.getvalue()

# python image.py --instruction_file=./example/instruction.txt
def main():
    parser = argparse.ArgumentParser(description="Generate 970x600 Amazon A+ module using OpenAI gpt-image-1.")
    parser.add_argument("--instruction_file", required=True, help="Path to text file containing design instructions or copy.")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"), help="Your OpenAI API key.")
    parser.add_argument(
        "--result_dir",
        default=None,
        help="Directory where run artifacts (prompt, intermediate images) are stored. "
        "Defaults to timestamped folder within ./results.",
    )
    args = parser.parse_args()

    if not args.api_key:
        raise SystemExit("❌ Missing OPENAI_API_KEY. Set it in your environment or pass --api_key.")

    start = time.time()
    client = OpenAI(api_key=args.api_key)

    # 1️⃣ Read instruction file
    with open(args.instruction_file, "r", encoding="utf-8") as f:
        instruction_text = f.read().strip()

    # 2️⃣ Build the prompt
    prompt = f"""{SYSTEM_MESSAGE}

Text content to include:
{instruction_text}

Design requirements:
- Focus on refined typography, subtle gradients, and premium color palettes.
- Use text as the hero element; product photography is optional but exclude logos.
- Keep ample white space and a clear hierarchy (headline, subhead, supporting copy).
- Ensure the overall design feels modern, calm, and brand-agnostic.
- Maintain the 970x600 aspect ratio (generated at 1536x1024 then downscaled)."""

    # 3️⃣ Prepare result directory & save prompt
    default_root = os.path.abspath("results")
    os.makedirs(default_root, exist_ok=True)

    run_id = time.strftime("%Y%m%d_%H%M%S")
    if args.result_dir:
        run_folder = os.path.abspath(args.result_dir)
        os.makedirs(run_folder, exist_ok=True)
    else:
        run_folder = os.path.join(default_root, run_id)
        while os.path.exists(run_folder):
            run_id = f"{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
            run_folder = os.path.join(default_root, run_id)
        os.makedirs(run_folder, exist_ok=True)

    prompt_path = os.path.join(run_folder, "prompt.txt")
    with open(prompt_path, "w", encoding="utf-8") as prompt_file:
        prompt_file.write(prompt)

    # 4️⃣ Generate text-focused image directly
    print("🎨 Generating base image (1536x1024)...")
    result = client.images.generate(
        model="gpt-image-1",
        prompt=prompt,
        size=f"{GEN_W}x{GEN_H}",
    )

    # 5️⃣ Save raw output, then downscale to 970x600
    b64 = result.data[0].b64_json
    raw_out_path = os.path.join(run_folder, "ai_output_1536x1024.png")
    save_b64_to_file(b64, raw_out_path)

    with Image.open(raw_out_path) as img:
        ai_image = img.convert("RGBA")
        final_resized = ai_image.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        final_out = os.path.join(run_folder, "final_output_970x600.png")
        final_resized.save(final_out)

    print(f"🗂️ Result artifacts stored in: {run_folder}")
    print(f"📝 Prompt saved to: {prompt_path}")
    print(f"🧠 Raw AI output (1536x1024): {raw_out_path}")
    print(f"✅ Final image (970x600): {final_out}")
    print("📏 Generated at 1536x1024 and downscaled to exact target.")
    end = time.time()
    print(f"Runtime: {end - start:.4f} seconds")

if __name__ == "__main__":
    main()
