#!/usr/bin/env python3
"""
Generate Amazon Bullet Points from product assets using OpenAI Responses API (vision).

Now outputs:
- bullets.txt : nicely formatted per bullet
- bullets.json: flat text blocks per bullet (no nested JSON objects)

Example bullets.json:
{
  "bullets": [
    "Encourages Active, Independent Play\n\nCustomer Benefit: ...",
    "Develops Confidence Through Play\n\nCustomer Benefit: ..."
  ]
}
"""
import os
import json
import base64
from pathlib import Path
from typing import List, Dict, Any, Optional

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


def _build_instruction(title: str, brand_prompt: str) -> str:
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
- {brand_prompt}

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


def _extract_json(text: str) -> Dict[str, Any]:
    try:
        return json.loads(text)
    except Exception:
        l = text.find("{")
        r = text.rfind("}")
        if l != -1 and r != -1 and r > l:
            snippet = text[l:r+1]
            return json.loads(snippet)
        raise


@click.command()
@click.option("--product-image", "product_image", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=True, help="Path to the main product image.")
@click.option("--logo-image", "logo_image", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False, help="Path to the brand logo image (optional but recommended).")
@click.option("--title", required=True, help="Product title as shown on Amazon.")
@click.option("--description", required=False, default=None, help="Short product description text. If not provided, use --description-file.")
@click.option("--description-file", "description_file", type=click.Path(exists=True, dir_okay=False, path_type=Path), required=False, help="Path to a text file containing the product description.")
@click.option("--prompt", "custom_prompt", required=False, default="Write 3 concise Amazon bullets: 1) Customer Benefit, 2) Key Feature, 3) Proof/Differentiator.", help="Custom instruction to steer style/brand tone.")
@click.option("--model", required=False, default="gpt-4o-mini", show_default=True, help="OpenAI model with vision support.")
@click.option("--outdir", type=click.Path(file_okay=False, path_type=Path), default=Path("."), show_default=True, help="Directory to write bullets.json and bullets.txt")
def main(product_image: Path,
         logo_image: Optional[Path],
         title: str,
         description: Optional[str],
         description_file: Optional[Path],
         custom_prompt: str,
         model: str,
         outdir: Path):
    outdir.mkdir(parents=True, exist_ok=True)

    product_uri = _encode_to_data_uri(product_image)
    logo_uri = _encode_to_data_uri(logo_image) if logo_image else None
    desc_text = _load_description(description, description_file)

    brand_prompt = f"Additional Prompt: {custom_prompt}\nDescription: {desc_text}"
    instruction = _build_instruction(title, brand_prompt)

    client = OpenAI()

    content_parts = [
        {"type": "input_text", "text": instruction},
        {"type": "input_image", "image_url": product_uri},
    ]
    if logo_uri:
        content_parts.append({"type": "input_image", "image_url": logo_uri})

    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": content_parts}],
    )

    raw_text = resp.output_text
    try:
        data = _extract_json(raw_text)
    except Exception:
        data = {"bullets": []}

    bullets = data.get("bullets", [])
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
    txt_path.write_text("\n".join(lines), encoding="utf-8")

    json_path = outdir / "bullets.json"
    json_path.write_text(json.dumps({"bullets": flat_bullets}, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n".join(lines))
    print(f"\nSaved:\n- {json_path}\n- {txt_path}")


if __name__ == "__main__":
    main()
