# Amazon A+ Content Generator

This project generates Amazon A+ Content modules using OpenAI’s image generation API.  
It takes in a product image, brand logo, and product details, then produces optimized image modules for A+ listings.

---

## 🧩 Prerequisites

- **Python 3.10+** installed
- **OpenAI API key** set in your environment (e.g. `export OPENAI_API_KEY=your_key_here`)

Or create a .env under your dev folder
add the following line
```bash
OPENAI_API_KEY=your_key_here
```
run
```bash
source .env
```
---

## 🐍 Setup Instructions

### 1. Create a Python Virtual Environment

```bash
python3 -m venv venv
```

### 2. Activate the Virtual Environment

- **macOS/Linux:**
  ```bash
  source venv/bin/activate
  ```

- **Windows (PowerShell):**
  ```bash
  .\venv\Scripts\activate
  ```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

## 🚀 Running the CLI

You can generate the A+ content by running the following command:

```bash
python main.py   --product-image="./example/product.jpg"   --logo-image="./example/logo.jpg"   --title="Tiny Land Triangle Climbing Frame for Kids, 3 in 1 Foldable Montessori Toy Wooden Climbing Triangle, Indoor Climbing Frame Set Wooden Climbing Toy with Ramp and Arch Climbing Triangle"   --outdir="./test2"
```
or
```bash
python main.py --product-image="./example/product.jpg" --logo-image="./example/logo.jpg" --title-path="./example/title.txt" --outdir="./test"
```

---

## 📂 Output

The generated modules and related metadata (JSON/text/images) will be saved to the folder specified in `--outdir`.

---

## 🧰 Notes

- Make sure your product and logo image files exist in the provided paths.
- You can customize the CLI options or extend the script to handle multiple product inputs.
- For debugging, use the `--verbose` flag if implemented.


