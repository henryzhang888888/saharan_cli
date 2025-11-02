#!/usr/bin/env python3

import json
import io
import os
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional
from PIL import Image
import time
import click
from openai import OpenAI

def _encode_to_data_uri(image_path: Path) -> str:
    ext = image_path.suffix.lower().replace(".", "")
    if ext in {"jpg", "jpeg"}:
        mime = "image/jpeg"
    elif ext in {"png"}:
        mime = "image/png"
    elif ext in {"webp"}:
        mime = "image/webp"
    else:
        mime = "application/octet-stream"
    b64 = base64.b64encode(image_path.read_bytes()).decode("utf-8")
    return f"data:{mime};base64,{b64}"


def _load_description(description: Optional[str], description_file: Optional[Path]) -> str:
    if description and description.strip():
        return description.strip()
    if description_file and description_file.exists():
        return description_file.read_text(encoding="utf-8").strip()
    return ""


def _build_bullet_llm_instruction(title: str) -> str:
    return f"""\
You are an Amazon E-commerce Content Manager.
Task: Create **exactly 3** structured bullet items based on the product title, textual description, and the provided images (product + brand logo).

For EACH bullet item, produce:
- heading: a short hook (2–6 words) that encapsulates the benefit/idea
- customer_benefit: the shopper outcome (150–220 chars)
- key_feature: what it is / how it works (150–220 chars)
- proof_or_differentiator: materials, safety, certifications, brand detail, ratings, unique design, warranty, etc. (150–220 chars)

Style & Compliance:
- Avoid emojis, ALL CAPS blocks, and promotional language ("best", "No.1"). 
- No medical or unsafe claims. Don’t invent certifications or guarantees.
- Use concise, benefit-first phrasing. Prefer active voice.
- If any claim is uncertain from inputs, use safe, generic phrasing (e.g., "durable wood construction").

Inputs:
- Title: {title}

OUTPUT FORMAT (JSON only):
{{
  "bullets": [
    {{
      "heading": "<2–6 word hook>",
      "customer_benefit": "<150–220 chars>",
      "key_feature": "<150–220 chars>",
      "proof_or_differentiator": "<150–220 chars>"
    }},
    ...
  ]
}}
"""

def _get_llm_json_response(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            snippet = text[l:r+1]
            return json.loads(snippet)
        raise

def _get_bullet_llm_prompt(title: str, product_uri: str, logo_uri: str ) -> List[Dict[str, Any]]:
    bullet_instruction = _build_bullet_llm_instruction(title)
    content_parts = [
        {"type": "input_text", "text": bullet_instruction},
        {"type": "input_image", "image_url": product_uri},
    ]

    if logo_uri:
        content_parts.append({"type": "input_image", "image_url": logo_uri})

    return [{"role": "user", "content": content_parts}]

def _build_module_llm_instruction(title: str, bullets: str) -> str:
    module_user_prompt = """
Using the product’s image, logo, title, description, and bullet points,
design five (5) optimized Amazon A+ Content modules (970 × 600 px each) that: 
* Reflect a unified visual identity aligned with the brand’s tone and style. 
* Clearly showcase the product image in every module (specify exact placement).
* Maintain a consistent color palette, typography, and tone across all modules.
 
 Design Requirements For each module, specify the following:
 1. Purpose & Layout – Define the objective (awareness, education, proof, conversion) and describe the layout (e.g., split-left visual, icon grid, hero image).
 2. Background & Visual Theme – Outline background colors, textures, or imagery (include color codes if applicable). 
 3. Text & Alignment – Provide exact text (headline, captions, supporting copy, CTA) and specify alignment (left, right, center).
 4. Design Tone & Style – Define the overall tone (e.g., premium, modern, lifestyle-driven) and ensure consistent typography (e.g., Headline: Poppins SemiBold 36pt; Body: Open Sans Regular 20pt).
 5. Product Placement – Indicate how and where the product appears (e.g., hero shot, lifestyle setting, close-up).
 6. Brand Logo Usage – Include the brand logo appropriately without alteration or cropping. Exclude this requirement if brand logo is not provided 
 
Narrative Flow Propose the sequence of modules (1–5) and explain the storytelling logic, for example
Module 1: Brand Story (Awareness) → Module 2: Craftsmanship (Education) → Module 3: Features Breakdown (Proof) → Module 4: Lifestyle & Use Cases (Connection) → Module 5: Reviews & CTA (Conversion).

Each module should serve a unique purpose while contributing to a cohesive and engaging overall narrative.
"""

    inputs_block = (
        f"\n\nInputs for context:\n"
        f"Product Title:\n{title}\n\n"
        f"Bullet Points:{bullets}\n"
        f"\nAssets:\n- Product image (data URI provided)\n- Brand logo (data URI provided)\n"
    )

    outputs_block = """
OUTPUT FORMAT (JSON only):
{{
  "modules": [
    {{
      "type": "<Module Type>",
      "title": "<Module Title>",
      "visual_concept": "<2–4 concise sentences describing the visual scene, composition, and where the product image and logo appear. Assume a 970×600 px canvas. Do not invent assets.>",
      "headline": "<Short, benefit-driven headline>",
      "subtext": "<2–4 concise sentences in brand voice: premium, modern, calm, Montessori-inspired. Avoid unverifiable claims.>",
    }},
    ...
  ]
}}
"""
    return module_user_prompt + inputs_block + outputs_block

def _get_module_llm_prompt(title: str, bullets: str, product_uri: str, logo_uri: str ) -> List[Dict[str, Any]]:
    module_system_prompt = """You are a senior Amazon A+ Content designer.
You must output EXACTLY five modules in JSON.

Rules:
- Output five modules.
- Maintain a unified identity across all modules (palette #F8F3EC base, #BFD8B8 accent, text #2A2A2A; typography: Poppins SemiBold for headlines, Open Sans for body). You do NOT need to restate fonts/colors in the output.
- The product image and brand logo must be referenced for placement in EVERY module (e.g., “product shown center-left,” “logo bottom-right”).
- Avoid hard CTAs except Module 5 may nudge toward purchase subtly.
- Do NOT include any text outside the five modules. No preamble or epilogue.
"""

    module_instruction = _build_module_llm_instruction(title, bullets)
    user_content_parts = [
        {"type": "text", "text": module_instruction},
        {"type": "image_url", "image_url": {"url": product_uri}},
    ]

    if logo_uri:
        user_content_parts.append({"type": "image_url", "image_url": {"url":logo_uri}})

    return [
        {"role": "system", "content": module_system_prompt},
        {
            "role": "user",
            "content": user_content_parts
        }
    ]

def stringify_module(obj: dict) -> str:
    return (
        f'{obj["type"]}: “{obj["title"]}”\n\n'
        f'Visual Concept:\n{obj["visual_concept"]}\n\n'
        f'Headline:\n“{obj["headline"]}”\n\n'
        f'Subtext:\n{obj["subtext"]}'
    )

def _build_image_prompt(module_instruction: str) -> str:
    image_system_message = (
        "Create a 970:600 px full module (strictly follow this size for the generated module, "
        "don’t generate other sizes) for the product following the instructions below.\n"
        "Make sure to read the instructions carefully and include all required text without missing any details.\n"
        "Don’t change the look of the submitted logo if included."
    )

    return f"""{image_system_message}

Instructions:
{module_instruction}

Placement guidelines:
- Include both the provided PRODUCT and LOGO in the final image.
- Keep the LOGO pristine (no visual alterations); place it cleanly (e.g., top-right).
- Display the PRODUCT prominently (left or center-left) with space for text on the right.
- Maintain a cohesive, premium, Montessori-inspired tone and color harmony.
"""

# Target Amazon A+ spec
TARGET_W, TARGET_H = 970, 600
# Closest valid OpenAI generation size
GEN_W, GEN_H = 1536, 1024

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

def generate_image(client: OpenAI, product_image_path: str, logo_image_path: str, instruction: str, idx: int, outdir: str):
    base_canvas = make_blank_canvas_png(GEN_W, GEN_H)

    # 4️⃣ Generate using supported size 1536x1024
    print(f"🎨 Generating base image (1536x1024) for {idx}...")
    result = client.images.edit(
        model="gpt-image-1",
        prompt=_build_image_prompt(instruction),
        size=f"{GEN_W}x{GEN_H}",
        image=[
            ("canvas.png", base_canvas),
            (os.path.basename(product_image_path), open(product_image_path, "rb").read()),
            (os.path.basename(logo_image_path), open(logo_image_path, "rb").read()),
        ],
    )

    # 5️⃣ Save and downscale to 970x600
    b64 = result.data[0].b64_json
    tmp_path = Path(outdir) / f"_tmp_generated_{idx}.png"
    save_b64_to_file(b64, tmp_path)

    out_image_path = Path(outdir) / f"image_{idx}.png"
    with Image.open(tmp_path) as img:
        final = img.resize((TARGET_W, TARGET_H), Image.LANCZOS)
        final.save(out_image_path)

    print(f"✅ Saved final image: {out_image_path} (970x600)")

    tmp_path.unlink()

@click.command()
@click.option("--product-image", "product_image", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Path to the main product image.")
@click.option("--logo-image", "logo_image", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False, help="Path to the brand logo image (optional but recommended).")
@click.option("--title", required=False, help="Product title as shown on Amazon.")
@click.option("--title-path", "title_path", type=click.Path(exists=True), required=False, help="Path to a file containing the title")
@click.option("--model", required=False, default="gpt-4o-mini", show_default=True, help="OpenAI model with vision support.")
@click.option("--outdir", type=click.Path(file_okay=False, path_type=Path), default=Path("."), show_default=True, help="Directory to write bullets.json and bullets.txt")
def main(product_image: Path,
         logo_image: Optional[Path],
         title: str,
         title_path: str,
         model: str,
         outdir: Path):

    if not title and not title_path:
        raise click.UsageError("You must provide either --title or --title-path")

    if not title and title_path:
        with open(title_path, "r", encoding="utf-8") as f:
            title = f.read().strip()

    click.echo(f"Title: {title}")
    start = time.time()
    outdir.mkdir(parents=True, exist_ok=True)

    product_uri = _encode_to_data_uri(product_image)
    logo_uri = _encode_to_data_uri(logo_image) if logo_image else None

    client = OpenAI()

    print("Generating the bullets")
    bullets_llm_prompt = _get_bullet_llm_prompt(title, product_uri, logo_uri)
    resp = client.responses.create(
        model=model,
        input=bullets_llm_prompt,
    )

    print("Tokens for bullets:", json.dumps(resp.usage.model_dump(), indent=2))

    raw_text = resp.output_text
    try:
        bullets_data = _get_llm_json_response(raw_text)
    except Exception:
        bullets_data = {"bullets": []}

    bullets = bullets_data.get("bullets", [])
    def _blank_item():
        return {
            "heading": "Encourages Active, Independent Play",
            "customer_benefit": "Helps children build balance, confidence, and coordination through natural movement and exploration.",
            "key_feature": "Montessori-inspired 3-in-1 climbing set includes a foldable triangle, reversible ramp, and arch for endless play configurations.",
            "proof_or_differentiator": "Designed to support gross motor skill development while keeping kids active and engaged indoors — no screens required."
        }

    while len(bullets) < 3:
        bullets.append(_blank_item())
    bullets = bullets[:3]

    # Build TXT output (multi-line, same as before)
    lines = []
    flat_bullets = []  # for flat JSON format
    for idx, item in enumerate(bullets, start=1):
        text_block = (
            f"{item.get('heading','').strip()}\n\n"
            f"Customer Benefit: {item.get('customer_benefit','').strip()}\n"
            f"Key Feature: {item.get('key_feature','').strip()}\n"
            f"Proof / Differentiator: {item.get('proof_or_differentiator','').strip()}"
        )
        flat_bullets.append(text_block)

        # For txt file with numbering
        lines.append(f"{idx}. {item.get('heading','').strip()}")
        lines.append(f"Customer Benefit: {item.get('customer_benefit','').strip()}")
        lines.append(f"Key Feature: {item.get('key_feature','').strip()}")
        lines.append(f"Proof / Differentiator: {item.get('proof_or_differentiator','').strip()}")
        if idx < len(bullets):
            lines.append("")

    txt_path = outdir / "bullets.txt"
    bullets_txt = "\n".join(lines)
    txt_path.write_text(bullets_txt, encoding="utf-8")

    json_path = outdir / "bullets.json"
    json_path.write_text(json.dumps({"bullets": flat_bullets}, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"\nBullets Saved:\n- {json_path}\n- {txt_path}")

    print("generating module......")
    module_llm_prompt=_get_module_llm_prompt(title, bullets_txt, product_uri, logo_uri)
    module_completion = client.chat.completions.create(
        model=model,
        messages=module_llm_prompt,
        temperature=1,
    )

    print("Tokens for modules:", json.dumps(module_completion.usage.model_dump(), indent=2))

    module_content = (module_completion.choices[0].message.content or "").strip()
    try:
        modules_data = _get_llm_json_response(module_content)
    except Exception:
        modules_data = {"modules": []}

    modules = modules_data.get("modules", [])

    text_modules = []
    for module in modules:
        formatted = stringify_module(module)
        text_modules.append(formatted)

    module_txt_path = outdir / "modules.txt"

    with open(module_txt_path, "w", encoding="utf-8") as f:
        f.write("\n".join(text_modules) + "\n")

    click.echo(f"✅ Modules written to {module_txt_path}")

    print("Generating image")
    for current_idx, current_text_module in enumerate(text_modules):
        generate_image(client, product_image, logo_image, current_text_module, current_idx, outdir)

    end = time.time()
    print(f"Runtime: {end - start:.4f} seconds")

if __name__ == "__main__":
    main()
