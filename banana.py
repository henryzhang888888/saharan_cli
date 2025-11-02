import os
import time
from io import BytesIO
from PIL import Image
import click
from google import genai
import time

SYSTEM_MESSAGE = (
    "Create a 970:600 px full module (strictly follow this size for the generated module, "
    "don’t generate other sizes) for the product following the instructions below.\n"
    "Make sure to read the instructions carefully and include all required text without missing any details.\n"
    "Don’t change the look of the submitted logo if included."
)

FINAL_WIDTH = 970
FINAL_HEIGHT = 600

def read_instruction_file(path: str) -> str:
    """Read generation instruction from a local text file."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"❌ Instruction file not found: {path}")
    with open(path, "r", encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        raise ValueError("❌ Instruction file is empty.")
    return content


def save_inline_image(part, outdir, stem="gemini_result"):
    """Save inline image returned from Gemini."""
    os.makedirs(outdir, exist_ok=True)
    data = part.inline_data.data
    img = Image.open(BytesIO(data))
    resized = img.resize((FINAL_WIDTH, FINAL_HEIGHT), Image.LANCZOS)

    ts = int(time.time())
    path = os.path.join(outdir, f"{stem}_{ts}.png")
    resized.save(path)
    return path

SUPPORTED_SIZES = {"1024x1024", "1024x1536", "1536x1024", "auto"}

def generate_image(product_path: str, logo_path: str, instruction_path: str, outdir:str, size: str):
    """Generate a composite image using Gemini."""
    if not os.path.exists(product_path):
        raise FileNotFoundError(f"❌ Missing product image: {product_path}")
    if not os.path.exists(logo_path):
        raise FileNotFoundError(f"❌ Missing logo image: {logo_path}")

    instruction_text = read_instruction_file(instruction_path)
    client = genai.Client()

    prompt = f"""{SYSTEM_MESSAGE}

Instructions:
{instruction_text}

Placement guidelines:
- Include both the provided PRODUCT and LOGO in the final image.
- Keep the LOGO pristine (no visual alterations); place it cleanly (e.g., top-right).
- Display the PRODUCT prominently (left or center-left) with space for text on the right.
- Maintain a cohesive, premium, Montessori-inspired tone and color harmony.
"""
    # Load images
    product_img = Image.open(product_path)
    logo_img = Image.open(logo_path)

    # === Generate ===
    response = client.models.generate_content(
        model="gemini-2.5-flash-image",
        contents=[prompt, product_img, logo_img],
        config=genai.types.GenerateContentConfig(
            response_modalities=[
                genai.types.Modality.TEXT,
                genai.types.Modality.IMAGE,
            ],
            candidate_count=1,
            image_config=genai.types.ImageConfig(aspect_ratio="4:3")
        ),
    )

    image_paths = []
    for cand in response.candidates:
        for part in cand.content.parts:
            if getattr(part, "inline_data", None) and part.inline_data.mime_type.startswith("image/"):
                image_paths.append(save_inline_image(part, outdir))
            elif getattr(part, "text", None):
                click.echo(click.style(f"Model note: {part.text}", fg="yellow"))

    if not image_paths:
        click.echo(click.style("⚠️ No images returned. Check safety filters or instruction content.", fg="red"))
    else:
        click.echo(click.style("✅ Generated image(s):", fg="green"))
        for p in image_paths:
            click.echo(f"- {p}")


@click.command()
@click.option(
    "--product-image",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the product image (e.g., ./product.jpg).",
)
@click.option(
    "--logo-image",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to the logo image (e.g., ./logo.jpg).",
)
@click.option(
    "--instruction-path",
    type=click.Path(exists=True, dir_okay=False),
    required=True,
    help="Path to text file containing generation instructions.",
)
@click.option(
    "--outdir",
    type=click.Path(file_okay=False),
    default="./outputs",
    show_default=True,
    help="Output directory for generated images.",
)
@click.option(
    "--size",
    type=click.Choice(list(SUPPORTED_SIZES)),
    default="1536x1024",
    show_default=True,
    help="Generation size (Gemini supports 1024x1024, 1024x1536, 1536x1024, or auto).",
)
def main(product_image, logo_image, instruction_path, size, outdir):
    """Generate a 970x600 Amazon A+ module image using Gemini."""
    start = time.time()
    generate_image(product_image, logo_image, instruction_path, outdir, size)
    end = time.time()
    print(f"\nRuntime: {end - start:.4f} seconds\n")

# python banana.py --product-image "./example/product.jpg" --logo-image "./example/logo.jpg" --instruction-path "./example/instruction.txt"  --outdir "./test_banana"
if __name__ == "__main__":
    main()

