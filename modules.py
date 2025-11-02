#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os, json, base64, mimetypes
from typing import List
import click
from openai import OpenAI

# ========= Fixed Output Requirements =========
# We enforce the exact, plain-text module template (no JSON, no markdown).

SYSTEM_PROMPT = """You are a senior Amazon A+ Content designer.
You must output EXACTLY five modules in plain text, no JSON, no markdown, and no extra commentary.
Follow this per-module template and headings verbatim:

Module N — <Module Type>: “<Module Title>”

Visual Concept:
<2–4 concise sentences describing the visual scene, composition, and where the product image and logo appear. Assume a 970×600 px canvas. Do not invent assets.>

Headline:
“<Short, benefit-led headline>”

Subtext:
<2–4 concise sentences in brand voice: premium, modern, calm, Montessori-inspired. Avoid unverifiable claims.>

Rules:
- Output five modules, numbered 1 to 5, each separated by a SINGLE blank line.
- Maintain a unified identity across all modules (palette #F8F3EC base, #BFD8B8 accent, text #2A2A2A; typography: Poppins SemiBold for headlines, Open Sans for body). You do NOT need to restate fonts/colors in the output.
- The product image and brand logo must be referenced for placement in EVERY module (e.g., “product shown center-left,” “logo bottom-right”).
- Avoid hard CTAs except Module 5 may nudge toward purchase subtly.
- Do NOT include any text outside the five modules. No preamble or epilogue.
"""

# ========= Your EXACT user prompt (verbatim) =========
FIXED_USER_PROMPT = """Using the product’s image, logo, title, description, and bullet points, design five (5) optimized Amazon A+ Content modules (970 × 600 px each) that: * Reflect a unified visual identity aligned with the brand’s tone and style. * Clearly showcase the product image in every module (specify exact placement). * Maintain a consistent color palette, typography, and tone across all modules. Design Requirements For each module, specify the following: 1. Purpose & Layout – Define the objective (awareness, education, proof, conversion) and describe the layout (e.g., split-left visual, icon grid, hero image). 2. Background & Visual Theme – Outline background colors, textures, or imagery (include color codes if applicable). 3. Text & Alignment – Provide exact text (headline, captions, supporting copy, CTA) and specify alignment (left, right, center). 4. Design Tone & Style – Define the overall tone (e.g., premium, modern, lifestyle-driven) and ensure consistent typography (e.g., Headline: Poppins SemiBold 36pt; Body: Open Sans Regular 20pt). 5. Product Placement – Indicate how and where the product appears (e.g., hero shot, lifestyle setting, close-up). 6. Brand Logo Usage – Include the brand logo appropriately without alteration or cropping. Exclude this requirement if brand logo is not provided Narrative Flow Propose the sequence of modules (1–5) and explain the storytelling logic, for example: Module 1: Brand Story (Awareness) → Module 2: Craftsmanship (Education) → Module 3: Features Breakdown (Proof) → Module 4: Lifestyle & Use Cases (Connection) → Module 5: Reviews & CTA (Conversion). Each module should serve a unique purpose while contributing to a cohesive and engaging overall narrative."""

# We append a minimal guidance block to bind that instruction to your required output template.
CONSTRAINTS_BRIDGE = """\n\nIMPORTANT: Present the five modules ONLY in the following text template (no JSON/markdown):

Module 1 — <Module Type>: “<Module Title>”

Visual Concept:
...

Headline:
“...”

Subtext:
...

(Repeat the same structure for Modules 2–5.)\n"""

# ========= Helper functions =========

def img_to_data_url(path: str) -> str:
    if not os.path.isfile(path):
        raise click.UsageError(f"Image not found: {path}")
    mime, _ = mimetypes.guess_type(path)
    if not mime or not mime.startswith("image/"):
        raise click.UsageError(f"Not an image file: {path}")
    with open(path, "rb") as f:
        b64 = base64.b64encode(f.read()).decode("utf-8")
    return f"data:{mime};base64,{b64}"

def load_bullets(json_path: str) -> List[str]:
    if not os.path.isfile(json_path):
        raise click.UsageError(f"Bullets JSON not found: {json_path}")
    try:
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except json.JSONDecodeError as e:
        raise click.UsageError(f"Invalid JSON in {json_path}: {e}")
    if not isinstance(data, dict) or "bullets" not in data:
        raise click.UsageError("Bullets JSON must be an object with key 'bullets': {\"bullets\": [..]}")
    bullets = data["bullets"]
    if not isinstance(bullets, list) or not all(isinstance(x, str) and x.strip() for x in bullets):
        raise click.UsageError("'bullets' must be a list of non-empty strings.")
    return bullets

# ========= CLI =========

@click.command()
@click.option("--product_image", required=True, help="Path to the product image (context only; no image generation).")
@click.option("--logo_image", required=True, help="Path to the brand logo image (context only).")
@click.option("--title", required=True, help="Product title.")
@click.option("--description", required=True, help="Product description (short paragraph).")
@click.option("--bullets_json", required=True, help='Path to JSON with {"bullet_points":[...]}')
@click.option("--model", default="gpt-5", show_default=True, help="OpenAI model.")
@click.option("--out", default="modules.txt", show_default=True, help="Where to write the five-module text.")
def main(product_image, logo_image, title, description, bullets_json, model, out):
    # Load inputs
    bullets = load_bullets(bullets_json)
    product_data_url = img_to_data_url(product_image)
    logo_data_url = img_to_data_url(logo_image)

    # Build the user message: your prompt verbatim + a compact inputs block
    bullets_lines = "\n- " + "\n- ".join(bullets)
    inputs_block = (
        f"\n\nInputs for context:\n"
        f"Product Title:\n{title}\n\n"
        f"Product Description:\n{description}\n\n"
        f"Bullet Points:{bullets_lines}\n"
        f"\nAssets:\n- Product image (data URI provided)\n- Brand logo (data URI provided)\n"
    )

    user_prompt = FIXED_USER_PROMPT + CONSTRAINTS_BRIDGE + inputs_block

    client = OpenAI()
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": user_prompt},
                    {"type": "image_url", "image_url": {"url": product_data_url}},
                    {"type": "image_url", "image_url": {"url": logo_data_url}},
                ],
            },
        ],
        temperature=1,
    )

    content = (completion.choices[0].message.content or "").strip()

    # Minimal format check
    if not content.startswith("Module 1"):
        with open(out + ".raw.txt", "w", encoding="utf-8") as f:
            f.write(content)
        raise click.UsageError(
            f"Unexpected output format. Saved raw output to {out}.raw.txt"
        )

    with open(out, "w", encoding="utf-8") as f:
        f.write(content + "\n")

    click.echo(f"✅ Modules written to {out}")

if __name__ == "__main__":
    main()
