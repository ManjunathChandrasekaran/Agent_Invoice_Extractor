import re
import json
import configparser
import os
from typing import Dict, List, Optional
from datetime import datetime
from dateutil.parser import parse
from llama_cpp import Llama
import pdfplumber
from PIL import Image
import pytesseract
from googletrans import Translator

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


class InvoiceExtractor:
    def __init__(self, model_path: str, n_ctx: int = 2048):
        """
        Initialize the InvoiceExtractor with the Llama 3.1 model.
        """
        try:
            self.llm_model = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=8)
            self.n_ctx = n_ctx
        except Exception as e:
            print(f"Failed to load LLM model: {e}. Falling back to rule-based extraction.")
            self.llm_model = None
            self.n_ctx = 0
        self.date_formats = [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # e.g., 12/31/2023 or 12-31-23
            r'\d{1,2}\s+[A-Za-z]+\s+\d{4}',  # e.g., 31 December 2023
        ]
        self.currency_pattern = r'[\$€£¥]\s*\d+(?:\.\d{2})?'  # Basic currency match

    def extract_with_llm(self, text: str) -> Dict:
        """
        Extract invoice data using the Llama 3.1 model.
        """
        if not self.llm_model:
            return self.extract_rule_based(text)

        # Truncate text if too long (estimate 4 chars per token)
        max_text_length = int(self.n_ctx * 0.75)  # Reserve 25% for prompt
        if len(text) > max_text_length:
            print(f"Input text too long ({len(text)} chars). Truncating to {max_text_length} chars.")
            text = text[:max_text_length]

        prompt = f"""
        <|begin_of_text|><|start_header_id|>system<|end_header_id>
        You are an expert in extracting structured data from invoices. Given the invoice text below, extract the following information and return it as a valid JSON object enclosed in ```json``` and `````` markers. Ensure the JSON is properly formatted and contains no syntax errors. If a field cannot be extracted, set it to null. Do not include any explanations or additional text outside the JSON.

        Fields to extract:
        - invoice_number: string (e.g., INV-12345)
        - invoice_date: string in YYYY-MM-DD format (e.g., 2023-12-31)
        - vendor_name: string (e.g., Acme Corp)
        - vendor_address: string (e.g., 123 Main St, Springfield, IL 62701)
        - line_items: list of objects with description (string), quantity (integer), unit_price (float), total (float)
        - total_amount: float (e.g., 35.00)
        - currency: string (e.g., USD, EUR)

        Invoice Text:
        {text}

        Output format:
        ```json
        {{
          "invoice_number": null,
          "invoice_date": null,
          "vendor_name": null,
          "vendor_address": null,
          "line_items": [],
          "total_amount": null,
          "currency": null
        }}
        ```
        <|end_header_id><|eot_id><|start_header_id|>user<|end_header_id>
        {text}
        <|eot_id><|start_header_id|>assistant<|end_header_id>
        """
        try:
            response = self.llm_model(prompt, max_tokens=512, stop=["<|eot_id|>"], temperature=0.3)
            output = response["choices"][0]["text"].strip()
            # Log raw output for debugging
            print(f"Raw LLM output:\n{output}\n")
            # Try extracting JSON from ```json``` blocks
            json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
            if json_match:
                return json.loads(json_match.group(1))
            # Fallback: Try parsing output as JSON directly
            try:
                return json.loads(output)
            except json.JSONDecodeError:
                # Clean output and try again
                cleaned_output = output.strip().replace('\n', ' ').strip('`')
                try:
                    return json.loads(cleaned_output)
                except json.JSONDecodeError as e:
                    print(f"No valid JSON found in LLM output: {e}. Falling back to rule-based extraction.")
                    return self.extract_rule_based(text)
        except Exception as e:
            print(f"LLM extraction failed: {e}. Falling back to rule-based extraction.")
            return self.extract_rule_based(text)

    def extract_rule_based(self, text: str) -> Dict:
        """
        Rule-based extraction as a fallback or standalone method.
        """
        result = {
            "invoice_number": None,
            "invoice_date": None,
            "vendor_name": None,
            "vendor_address": None,
            "line_items": [],
            "total_amount": None,
            "currency": None
        }

        # Normalize text
        text = text.lower().replace('\n', ' ').strip()

        # Extract invoice number
        invoice_num_match = re.search(r'invoice\s*(?:#|no|number)\s*[:\-]?\s*([a-z0-9\-]+)', text)
        if invoice_num_match:
            result["invoice_number"] = invoice_num_match.group(1)

        # Extract date
        for date_pattern in self.date_formats:
            date_match = re.search(date_pattern, text)
            if date_match:
                try:
                    parsed_date = parse(date_match.group(0), fuzzy=True)
                    result["invoice_date"] = parsed_date.strftime('%Y-%m-DD')
                    break
                except:
                    continue

        # Extract vendor name (heuristic: look for "from", "vendor", or company-like names)
        vendor_match = re.search(r'(?:from|vendor|supplier)\s*[:\-]?\s*([a-z\s\&]+)', text)
        if vendor_match:
            result["vendor_name"] = vendor_match.group(1).strip().title()

        # Extract vendor address (heuristic: look for street, city, zip patterns)
        address_match = re.search(r'\d+\s+[a-z\s]+,\s*[a-z\s]+,\s*[a-z]{2}\s*\d{5}', text)
        if address_match:
            result["vendor_address"] = address_match.group(0).title()

        # Extract line items (heuristic: look for patterns like qty, description, price)
        line_item_pattern = r'(\d+)\s+([a-z\s]+?)\s+(\d+\.\d{2})\s+(\d+\.\d{2})'
        line_items = re.findall(line_item_pattern, text)
        for qty, desc, unit_price, total in line_items:
            result["line_items"].append({
                "description": desc.strip().title(),
                "quantity": int(qty),
                "unit_price": float(unit_price),
                "total": float(total)
            })

        # Extract total amount
        total_match = re.search(r'total\s*(?:amount)?\s*[:\-]?\s*([\$\€\£\¥]?\s*\d+\.\d{2})', text)
        if total_match:
            amount = total_match.group(1)
            result["total_amount"] = float(re.sub(r'[\$\€\£\¥]', '', amount))
            if amount[0] in '$€£¥':
                result["currency"] = {'$': 'USD', '€': 'EUR', '£': 'GBP', '¥': 'JPY'}.get(amount[0], 'USD')

        return result

    def extract(self, text: str) -> Dict:
        """
        Main method to extract invoice data, preferring LLM if available.
        """
        if self.llm_model:
            return self.extract_with_llm(text)
        return self.extract_rule_based(text)


def extract_text_from_pdf(pdf_path: str) -> tuple[str, bool]:
    """
    Extract text from a PDF file using pdfplumber, with OCR fallback using pytesseract.
    Returns (text, is_scanned) where is_scanned indicates if OCR was used.
    """
    try:
        text = ""
        is_scanned = False
        with pdfplumber.open(pdf_path) as pdf:
            for page in pdf.pages:
                # Try text extraction first
                page_text = page.extract_text()
                if page_text and len(page_text.strip()) > 50:  # Arbitrary threshold for text-based PDF
                    text += page_text + " "
                else:
                    # Fallback to OCR for scanned PDFs
                    is_scanned = True
                    img = page.to_image().original
                    page_text = pytesseract.image_to_string(img)
                    text += page_text + " "
        text = text.strip()
        if not text:
            print("No text extracted from PDF (empty or OCR failed).")
        return text, is_scanned
    except Exception as e:
        print(f"Failed to extract text from PDF: {e}")
        return "", False


def translate_text(text: str) -> str:
    """
    Translate text to English using googletrans.
    """
    if not text:
        return text
    try:
        translator = Translator()
        translated = translator.translate(text, dest='en').text
        return translated.strip()
    except Exception as e:
        print(f"Translation failed: {e}. Using original text.")
        return text


def load_config(config_file: str = "config.ini") -> Dict[str, str]:
    """
    Load model_path and pdf_path from a .ini file.
    """
    config = configparser.ConfigParser()
    try:
        config.read(config_file)
        if 'Paths' not in config:
            raise KeyError("Section 'Paths' not found in config file")
        paths = {
            'model_path': config['Paths'].get('model_path', ''),
            'pdf_path': config['Paths'].get('pdf_path', '')
        }
        # Validate paths
        if not paths['model_path'] or not os.path.exists(paths['model_path']):
            raise FileNotFoundError(f"Model path '{paths['model_path']}' is invalid or does not exist")
        if not paths['pdf_path'] or not os.path.exists(paths['pdf_path']):
            raise FileNotFoundError(f"PDF path '{paths['pdf_path']}' is invalid or does not exist")
        return paths
    except Exception as e:
        print(f"Failed to load config file '{config_file}': {e}")
        return {}


def save_json_result(result: Dict, pdf_path: str, output_dir: str = "output") -> None:
    """
    Save the extracted data as a JSON file in the output directory.
    """
    try:
        os.makedirs(output_dir, exist_ok=True)
        # Use PDF filename (without extension) for JSON file
        pdf_filename = os.path.splitext(os.path.basename(pdf_path))[0]
        output_path = os.path.join(output_dir, f"{pdf_filename}_extracted.json")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        print(f"JSON result saved to: {output_path}")
    except Exception as e:
        print(f"Failed to save JSON result: {e}")


def main():
    # Load configuration
    config = load_config()
    if not config:
        print("Exiting due to invalid configuration.")
        return

    model_path = config['model_path']
    pdf_path = config['pdf_path']

    # Extract text from PDF
    invoice_text, is_scanned = extract_text_from_pdf(pdf_path)
    if not invoice_text:
        print("No text extracted from PDF. Exiting.")
        return

    # Log whether PDF is scanned
    print(f"PDF is {'scanned (using OCR)' if is_scanned else 'text-based'}")
    print(f"Extracted PDF text:\n{invoice_text}\n")

    # Translate text to English
    invoice_text = translate_text(invoice_text)
    print(f"Translated text (English):\n{invoice_text}\n")

    # Initialize extractor with the Llama 3.1 model
    extractor = InvoiceExtractor(model_path=model_path, n_ctx=2048)

    # Extract data
    result = extractor.extract(invoice_text)

    # Output result to console
    print("Extracted invoice data:")
    print(json.dumps(result, indent=2))

    # Save result to output folder
    save_json_result(result, pdf_path)


if __name__ == "__main__":
    main()