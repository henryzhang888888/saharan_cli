import os
import io
import base64
import argparse
from PIL import Image
from openai import OpenAI
import time

# Target Amazon A+ spec
TARGET_W, TARGET_H = 970, 600
# Closest valid OpenAI generation size
GEN_W, GEN_H = 1536, 1024

SYSTEM_MESSAGE = (
    "Create a 970:600 px full module (strictly follow this size for the generated module, "
    "don’t generate other sizes) for the product following the instructions below.\n"
    "Make sure to read the instructions carefully and include all required text without missing any details.\n"
    "Don’t change the look of the submitted logo if included."
)

def make_blank_canvas_png(w: int, h: int) -> bytes:
    """Create transparent PNG canvas for edit API."""
    img = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()

def save_b64_to_file(b64: str, path: str):
    data = base64.b64decode(b64)
    with open(path, "wb") as f:
        f.write(data)

# python image.py --instruction_file=./example/instruction.txt --product_image=./example/product.jpg --logo_image=./example/logo.jpg
def main():
    parser = argparse.ArgumentParser(description="Generate 970x600 Amazon A+ module using OpenAI gpt-image-1.")
    parser.add_argument("--instruction_file", required=True, help="Path to text file containing design instructions.")
    parser.add_argument("--product_image", required=True, help="Path to product image (PNG/JPG).")
    parser.add_argument("--logo_image", required=True, help="Path to logo image (PNG/JPG).")
    parser.add_argument("--out", default="aplus_module_970x600.png", help="Output file name.")
    parser.add_argument("--api_key", default=os.getenv("OPENAI_API_KEY"), help="Your OpenAI API key.")
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

Instructions:
{instruction_text}

Placement guidelines:
- Include both the provided PRODUCT and LOGO in the final image.
- Keep the LOGO pristine (no visual alterations); place it cleanly (e.g., top-right).
- Display the PRODUCT prominently (left or center-left) with space for text on the right.
- Maintain a cohesive, premium, Montessori-inspired tone and color harmony.
"""

    # 3️⃣ Blank base canvas for edit
    base_canvas = make_blank_canvas_png(GEN_W, GEN_H)

    # 4️⃣ Generate using supported size 1536x1024
    print("🎨 Generating base image (1536x1024)...")
    result = client.images.edit(
        model="gpt-image-1",
        prompt=prompt,
        size=f"{GEN_W}x{GEN_H}",
        image=[
            ("canvas.png", base_canvas),
            (os.path.basename(args.product_image), open(args.product_image, "rb").read()),
            (os.path.basename(args.logo_image), open(args.logo_image, "rb").read()),
        ],
    )

    # 5️⃣ Save and downscale to 970x600
    b64 = result.data[0].b64_json
    tmp_path = "_tmp_generated.png"
    save_b64_to_file(b64, tmp_path)

    with Image.open(tmp_path) as img:
        final = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        final.save(args.out)

    print(f"✅ Saved final image: {args.out} (970x600)")
    print("📏 Generated at 1536x1024 and downscaled to exact target.")
    end = time.time()
    print(f"Runtime: {end - start:.4f} seconds")

if __name__ == "__main__":
    main()
