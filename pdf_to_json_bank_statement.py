"""
Enhanced PDF -> JSON converter for bank statements with column-wise OCR processing

Features:
- Column-wise OCR detection and text extraction
- Smart column mapping based on keywords and patterns
- Improved spatial analysis for better table structure detection
- Enhanced transaction parsing with column-specific validation
- Better handling of merged cells and complex layouts

Usage:
    python pdf_to_json_bank_statement.py input.pdf output.json [--ocr] [--password "pwd"]
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import os
import re
import sys
import getpass
from dataclasses import dataclass, asdict
from typing import List, Dict, Any, Optional, Tuple, NamedTuple

import pdfplumber
import pandas as pd
from dateutil import parser as dateparser
from PIL import Image
from pdfminer.pdfparser import PDFSyntaxError

# Optional OCR imports
try:
    import pytesseract
    from pdf2image import convert_from_path
    import cv2
    import numpy as np
    OCR_AVAILABLE = True
except Exception:
    OCR_AVAILABLE = False

# Enhanced Constants
CURRENCY_REGEX = re.compile(r"\b(INR|Rs\.?|₹|USD|EUR|GBP|JPY)\b", re.IGNORECASE)
AMOUNT_CLEAN = re.compile(r"[^0-9.\-]")
DATE_FINDER = re.compile(r"\d{1,2}[/-]\d{1,2}[/-]\d{2,4}")
BALANCE_PATTERN = re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*\((Cr|Dr)\)", re.IGNORECASE)
AMOUNT_PATTERN = re.compile(r"(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*\((Cr|Dr)\)", re.IGNORECASE)

# Header mapping for different bank formats
HEADER_MAPPINGS = {
    'date': ['date', 'dt', 'txn date', 'value date', 'transaction date', 'posting date', 'tran date'],
    'description': ['narration', 'description', 'particulars', 'details', 'transaction description', 'remarks', 'transaction details'],
    'reference': ['chq no', 'ref no', 'reference', 'chq/ref no', 'cheque no', 'transaction id', 'chq_ref_no', 'instrument no'],
    'amount': ['amount', 'withdrawal(dr)/deposit(cr)', 'debit/credit', 'withdrawal/deposit', 'dr/cr'],
    'debit': ['debit', 'withdrawal', 'dr', 'debited', 'withdraw', 'withdrawal amount'],
    'credit': ['credit', 'deposit', 'cr', 'credited', 'deposited', 'deposit amount'],
    'balance': ['balance', 'bal', 'running balance', 'closing balance', 'available balance']
}

# Column detection patterns and keywords
COLUMN_KEYWORDS = {
    'date': {
        'keywords': ['date', 'dt', 'txn date', 'value date', 'transaction date', 'posting date'],
        'patterns': [
            r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',
            r'\d{2}-\d{2}-\d{4}',
            r'\d{4}-\d{2}-\d{2}'
        ],
        'validation': lambda x: bool(re.search(r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}', str(x)))
    },
    'description': {
        'keywords': ['narration', 'description', 'particulars', 'details', 'transaction description', 'remarks'],
        'patterns': [
            r'^[A-Za-z].*',  # Starts with letter
            r'.*(?:transfer|payment|deposit|withdrawal|cheque|atm|pos|neft|rtgs|imps).*'
        ],
        'validation': lambda x: len(str(x).strip()) > 3 and not str(x).strip().isdigit()
    },
    'reference': {
        'keywords': ['chq no', 'ref no', 'reference', 'chq/ref no', 'cheque no', 'transaction id'],
        'patterns': [
            r'^[A-Z0-9]{6,}$',  # Alphanumeric reference
            r'^\d{6,}$',        # Numeric reference
            r'^CHQ\d+$',        # Cheque number
            r'^REF\d+$'         # Reference number
        ],
        'validation': lambda x: len(str(x).strip()) >= 4 and (str(x).strip().isdigit() or str(x).strip().isalnum())
    },
    'debit': {
        'keywords': ['debit', 'withdrawal', 'dr', 'debited', 'withdraw'],
        'patterns': [
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            r'\d+\.\d{2}'
        ],
        'validation': lambda x: is_valid_amount(x)
    },
    'credit': {
        'keywords': ['credit', 'deposit', 'cr', 'credited', 'deposited'],
        'patterns': [
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?',
            r'\d+\.\d{2}'
        ],
        'validation': lambda x: is_valid_amount(x)
    },
    'amount': {
        'keywords': ['amount', 'withdrawal(dr)/deposit(cr)', 'debit/credit'],
        'patterns': [
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*\((Cr|Dr)\)',
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
        ],
        'validation': lambda x: is_valid_amount(x) or bool(re.search(r'\d+.*\((Cr|Dr)\)', str(x)))
    },
    'balance': {
        'keywords': ['balance', 'bal', 'running balance', 'closing balance'],
        'patterns': [
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?\s*\((Cr|Dr)\)',
            r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?'
        ],
        'validation': lambda x: is_valid_amount(x) or bool(re.search(r'\d+.*\((Cr|Dr)\)', str(x)))
    }
}

class ColumnRegion(NamedTuple):
    x1: int
    y1: int
    x2: int
    y2: int
    column_type: str
    confidence: float

@dataclass
class CustomerInfo:
    name: Optional[str] = None
    account_number: Optional[str] = None
    statement_period: Optional[str] = None
    address: Optional[str] = None
    micr_code: Optional[str] = None
    ifsc_code: Optional[str] = None
    other: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        if self.other is None:
            self.other = {}

@dataclass
class Transaction:
    date: Optional[str]
    description: Optional[str]
    reference: Optional[str]
    type: Optional[str]
    amount: Optional[float]
    debit: Optional[float]
    credit: Optional[float]
    balance: Optional[float]
    raw: Dict[str, Any]

def is_valid_amount(text: str) -> bool:
    """Check if text contains a valid monetary amount"""
    if not text:
        return False
    text = str(text).strip()
    # Check for number patterns
    return bool(re.search(r'\d{1,3}(?:,\d{3})*(?:\.\d{2})?', text))

def detect_columns_ocr(image: np.ndarray, debug: bool = False) -> List[ColumnRegion]:
    """
    Detect table columns using OCR and spatial analysis
    """
    if not OCR_AVAILABLE:
        return []
    
    print("🔍 Detecting table columns using OCR...")
    
    # Convert PIL image to opencv format if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Get OCR data with bounding boxes
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    # Filter out low confidence text
    min_confidence = 30
    valid_indices = [i for i, conf in enumerate(ocr_data['conf']) if int(conf) > min_confidence]
    
    # Group text by approximate columns based on x-coordinates
    text_boxes = []
    for i in valid_indices:
        if ocr_data['text'][i].strip():
            text_boxes.append({
                'text': ocr_data['text'][i].strip(),
                'x': ocr_data['left'][i],
                'y': ocr_data['top'][i],
                'w': ocr_data['width'][i],
                'h': ocr_data['height'][i],
                'conf': ocr_data['conf'][i]
            })
    
    if not text_boxes:
        return []
    
    # Sort by y-coordinate to process rows
    text_boxes.sort(key=lambda x: x['y'])
    
    # Find header row (usually contains column names)
    header_candidates = []
    for box in text_boxes[:20]:  # Check first 20 text elements
        text_lower = box['text'].lower()
        for col_type, col_info in COLUMN_KEYWORDS.items():
            if any(keyword in text_lower for keyword in col_info['keywords']):
                header_candidates.append({
                    'box': box,
                    'column_type': col_type,
                    'confidence': 0.9
                })
                break
    
    print(f"📋 Found {len(header_candidates)} header candidates")
    
    # If no clear headers found, use pattern detection
    if len(header_candidates) < 3:
        print("🔎 Using pattern-based column detection...")
        header_candidates = []
        
        # Group text boxes by rows (similar y-coordinates)
        rows = []
        current_row = []
        last_y = -1
        y_threshold = 20  # pixels
        
        for box in text_boxes:
            if last_y == -1 or abs(box['y'] - last_y) <= y_threshold:
                current_row.append(box)
                last_y = box['y']
            else:
                if current_row:
                    rows.append(current_row)
                current_row = [box]
                last_y = box['y']
        
        if current_row:
            rows.append(current_row)
        
        # Analyze first few rows to detect column patterns
        for row_idx, row in enumerate(rows[:10]):
            # Sort boxes in row by x-coordinate
            row.sort(key=lambda x: x['x'])
            
            for box in row:
                text = box['text']
                for col_type, col_info in COLUMN_KEYWORDS.items():
                    # Check patterns
                    pattern_match = False
                    for pattern in col_info['patterns']:
                        if re.search(pattern, text, re.IGNORECASE):
                            pattern_match = True
                            break
                    
                    # Check validation
                    valid = False
                    try:
                        valid = col_info['validation'](text)
                    except:
                        valid = False
                    
                    if pattern_match or valid:
                        confidence = 0.8 if pattern_match and valid else 0.6
                        header_candidates.append({
                            'box': box,
                            'column_type': col_type,
                            'confidence': confidence
                        })
                        break
    
    # Create column regions
    columns = []
    
    # Sort header candidates by x-coordinate
    header_candidates.sort(key=lambda x: x['box']['x'])
    
    # Remove duplicates (same column type detected multiple times)
    seen_types = {}
    unique_candidates = []
    for candidate in header_candidates:
        col_type = candidate['column_type']
        if col_type not in seen_types or candidate['confidence'] > seen_types[col_type]['confidence']:
            seen_types[col_type] = candidate
    
    unique_candidates = list(seen_types.values())
    unique_candidates.sort(key=lambda x: x['box']['x'])
    
    print(f"📊 Detected {len(unique_candidates)} unique columns")
    
    # Create column regions with estimated boundaries
    image_height, image_width = image.shape[:2]
    
    for i, candidate in enumerate(unique_candidates):
        box = candidate['box']
        col_type = candidate['column_type']
        
        # Estimate column boundaries
        x1 = box['x']
        x2 = box['x'] + box['w']
        
        # Extend column width based on position
        if i > 0:
            prev_box = unique_candidates[i-1]['box']
            x1 = max(x1 - 20, prev_box['x'] + prev_box['w'])
        else:
            x1 = max(0, x1 - 30)
        
        if i < len(unique_candidates) - 1:
            next_box = unique_candidates[i+1]['box']
            x2 = min(next_box['x'], x2 + 50)
        else:
            x2 = min(image_width, x2 + 50)
        
        # Full height for now
        y1 = 0
        y2 = image_height
        
        column_region = ColumnRegion(
            x1=int(x1),
            y1=int(y1),
            x2=int(x2),
            y2=int(y2),
            column_type=col_type,
            confidence=candidate['confidence']
        )
        
        columns.append(column_region)
        
        print(f"   Column {col_type}: x({x1}-{x2}), confidence: {candidate['confidence']:.2f}")
    
    return columns

def extract_column_text_ocr(image: np.ndarray, column_regions: List[ColumnRegion]) -> List[Dict[str, Any]]:
    """
    Extract text from detected column regions
    """
    if not OCR_AVAILABLE or not column_regions:
        return []
    
    print(f"📝 Extracting text from {len(column_regions)} columns...")
    
    rows_data = []
    
    # Convert PIL image to opencv format if needed
    if isinstance(image, Image.Image):
        image = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
    
    # Get all text with coordinates
    ocr_data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
    
    # Filter and organize text boxes
    text_boxes = []
    for i, conf in enumerate(ocr_data['conf']):
        if int(conf) > 30 and ocr_data['text'][i].strip():
            text_boxes.append({
                'text': ocr_data['text'][i].strip(),
                'x': ocr_data['left'][i],
                'y': ocr_data['top'][i],
                'w': ocr_data['width'][i],
                'h': ocr_data['height'][i],
                'center_x': ocr_data['left'][i] + ocr_data['width'][i] // 2,
                'center_y': ocr_data['top'][i] + ocr_data['height'][i] // 2
            })
    
    # Group text boxes by rows
    text_boxes.sort(key=lambda x: x['y'])
    rows = []
    current_row = []
    last_y = -1
    y_threshold = 15
    
    for box in text_boxes:
        if last_y == -1 or abs(box['y'] - last_y) <= y_threshold:
            current_row.append(box)
            last_y = box['y']
        else:
            if current_row:
                rows.append(current_row)
            current_row = [box]
            last_y = box['y']
    
    if current_row:
        rows.append(current_row)
    
    print(f"📄 Found {len(rows)} text rows")
    
    # Extract data for each row
    for row_idx, row in enumerate(rows):
        if row_idx < 3:  # Skip likely header rows
            continue
            
        row_data = {'_row': row_idx, '_ocr_method': 'column_wise'}
        
        # For each column, find text that falls within its boundaries
        for column in column_regions:
            column_text = []
            
            for box in row:
                # Check if text box overlaps with column region
                if (column.x1 <= box['center_x'] <= column.x2 and 
                    column.y1 <= box['center_y'] <= column.y2):
                    column_text.append(box['text'])
            
            # Join text in column and clean
            if column_text:
                combined_text = ' '.join(column_text).strip()
                
                # Validate text for column type
                col_info = COLUMN_KEYWORDS.get(column.column_type, {})
                validation_func = col_info.get('validation')
                
                if validation_func:
                    try:
                        if validation_func(combined_text):
                            row_data[column.column_type] = combined_text
                    except:
                        row_data[column.column_type] = combined_text
                else:
                    row_data[column.column_type] = combined_text
        
        # Only add row if it has meaningful data
        if len([k for k in row_data.keys() if not k.startswith('_')]) >= 2:
            rows_data.append(row_data)
    
    print(f"✅ Extracted {len(rows_data)} data rows using column-wise OCR")
    
    # Debug: show first few rows
    for i, row in enumerate(rows_data[:3]):
        print(f"   Row {i}: {row}")
    
    return rows_data

def process_pdf_with_column_ocr(path: str, output_path: str, password: Optional[str] = None):
    """
    Process PDF using enhanced column-wise OCR
    """
    if not OCR_AVAILABLE:
        raise RuntimeError("OCR libraries not available for column-wise processing.")
    
    print("🚀 Starting column-wise OCR processing...")
    
    # Convert PDF to images
    try:
        images = convert_from_path(path, dpi=300)  # Higher DPI for better OCR
        print(f"📄 Converted {len(images)} pages to images")
    except Exception as e:
        print(f"❌ Failed to convert PDF to images: {e}")
        return
    
    all_transactions = []
    customer_info = CustomerInfo()
    bank_name = None
    images_b64 = []
    
    # Process each page
    for page_num, image in enumerate(images):
        print(f"\n📖 Processing page {page_num + 1}/{len(images)}")
        
        # Convert to numpy array for OpenCV processing
        img_array = cv2.cvtColor(np.array(image), cv2.COLOR_RGB2BGR)
        
        # Extract customer info from first page
        if page_num == 0:
            # Extract text for metadata
            full_text = pytesseract.image_to_string(image)
            customer_info = extract_customer_info(full_text)
            
            # Extract bank name
            lines = [line.strip() for line in full_text.splitlines() if line.strip()]
            for line in lines[:5]:
                if line and len(line) > 5 and not re.match(r'^[\d\s\-/]+$', line):
                    bank_name = line
                    break
            
            # Store base64 image
            images_b64.append(image_to_base64(image))
        
        # Detect columns
        column_regions = detect_columns_ocr(img_array, debug=True)
        
        if not column_regions:
            print(f"   ⚠️ No columns detected on page {page_num + 1}")
            continue
        
        # Extract column-wise text
        page_rows = extract_column_text_ocr(img_array, column_regions)
        
        if page_rows:
            print(f"   ✅ Extracted {len(page_rows)} rows from page {page_num + 1}")
            
            # Add page information
            for row in page_rows:
                row['_page'] = page_num + 1
            
            all_transactions.extend(page_rows)
    
    print(f"\n🎯 Total extracted rows: {len(all_transactions)}")
    
    # Normalize transactions
    normalized_transactions = normalize_transactions(all_transactions)
    
    # Guess currency
    currency = None
    if images:
        first_page_text = pytesseract.image_to_string(images[0])
        currency = guess_currency(first_page_text)
    
    # Build output structure
    output = build_output_structure(bank_name, customer_info, images_b64, normalized_transactions, currency)
    
    # Save to JSON
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False, indent=4)
    
    print(f"\n✅ Successfully saved JSON to {output_path}")
    print(f"📊 Found {len(normalized_transactions)} transactions")
    print(f"🏦 Bank: {bank_name or 'Unknown'}")
    print(f"👤 Customer: {customer_info.name or 'Unknown'}")
    print(f"💳 Account: {customer_info.account_number or 'Unknown'}")

def image_to_base64(pil_image: Image.Image) -> str:
    buf = io.BytesIO()
    pil_image.save(buf, format="PNG")
    return base64.b64encode(buf.getvalue()).decode("utf-8")

def open_pdf_with_password(path: str, password: Optional[str] = None):
    """Try to open a PDF with pdfplumber; prompt for password if needed."""
    try:
        if password:
            return pdfplumber.open(path, password=password)
        return pdfplumber.open(path)
    except (PDFSyntaxError, Exception) as e:
        while True:
            try:
                pwd = getpass.getpass("PDF appears to be password-protected. Enter password (leave empty to abort): ")
            except Exception:
                raise
            if pwd == "":
                raise RuntimeError("PDF is password protected and no password was provided.")
            try:
                return pdfplumber.open(path, password=pwd)
            except (PDFSyntaxError, Exception) as e:
                print("Password incorrect — try again.")

def extract_images_from_page(page) -> List[str]:
    images_b64: List[str] = []
    try:
        for image_obj in (page.images or []):
            try:
                x0 = image_obj.get("x0")
                top = image_obj.get("top")
                x1 = image_obj.get("x1")
                bottom = image_obj.get("bottom")
                if None not in (x0, top, x1, bottom):
                    im = page.within_bbox((x0, top, x1, bottom))
                    pil = im.to_image(resolution=150).original
                    images_b64.append(image_to_base64(pil))
                    continue
            except Exception:
                pass
            try:
                raw = image_obj.get("stream")
                if raw:
                    pil = Image.open(io.BytesIO(raw))
                    images_b64.append(image_to_base64(pil))
            except Exception:
                continue
    except Exception:
        pass
    return images_b64

def guess_currency(text: Optional[str]) -> Optional[str]:
    if not text:
        return None
    m = CURRENCY_REGEX.search(text)
    return m.group(1) if m else None

def clean_amount_with_brackets(s: Optional[Any]) -> Tuple[Optional[float], Optional[str]]:
    """Enhanced amount cleaning that handles (Cr)/(Dr) notation"""
    if s is None:
        return None, None
    
    s_str = str(s).strip()
    if not s_str:
        return None, None
    
    # Check for (Cr) or (Dr) pattern - more flexible matching
    patterns_to_try = [
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*\((Cr|Dr)\)',  # 1,200.00(Dr)
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*\(([CD]r)\)',  # 1,200.00(Dr) - case variations
        r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)\s*([CD]r)',      # 1,200.00Dr - without parentheses
    ]
    
    for pattern in patterns_to_try:
        balance_match = re.search(pattern, s_str, re.IGNORECASE)
        if balance_match:
            amount_str = balance_match.group(1).replace(',', '')
            cr_dr = balance_match.group(2).lower()
            try:
                amount = float(amount_str)
                tx_type = 'Credit' if cr_dr in ['cr', 'c'] else 'Debit'
                return amount, tx_type
            except ValueError:
                continue
    
    # Fallback to original logic for simple numbers
    if isinstance(s, (int, float)):
        return float(s), None
    
    # Try to extract just the number part
    s2 = AMOUNT_CLEAN.sub("", s_str.replace(",", ""))
    if s2 == "":
        return None, None
    
    try:
        amount = float(s2)
        tx_type = None
        if amount < 0:
            tx_type = "Debit"
            amount = abs(amount)
        return amount, tx_type
    except Exception:
        # Last resort: find any number in the string
        m = re.search(r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', s_str)
        if m:
            try:
                amount = float(m.group(1).replace(",", ""))
                # Determine type from context
                tx_type = None
                if 'dr' in s_str.lower() or 'debit' in s_str.lower():
                    tx_type = "Debit"
                elif 'cr' in s_str.lower() or 'credit' in s_str.lower():
                    tx_type = "Credit"
                return amount, tx_type
            except Exception:
                return None, None
    
    return None, None

def parse_date(s: Optional[str]) -> Optional[str]:
    if s is None:
        return None
    s = str(s).strip()
    if not s:
        return None
    
    try:
        # Try parsing with day-first assumption (common in Indian banks)
        dt = dateparser.parse(s, dayfirst=True, fuzzy=True)
        if dt:
            return dt.date().isoformat()
    except Exception:
        pass
    
    # Try extracting date pattern
    m = DATE_FINDER.search(s)
    if m:
        try:
            dt = dateparser.parse(m.group(0), dayfirst=True)
            if dt:
                return dt.date().isoformat()
        except Exception:
            pass
    
    return None

def extract_text_blocks(pdf, max_pages: int = 3) -> str:
    snippets: List[str] = []
    for page in pdf.pages[:max_pages]:
        try:
            text = page.extract_text() or ""
        except Exception:
            text = ""
        snippets.append(text)
    return "\n\n".join(snippets)

def extract_customer_info(text: str) -> CustomerInfo:
    info = CustomerInfo()
    if not text:
        return info
    
    lines = text.split('\n')
    
    # Account number patterns
    for line in lines:
        # MICR Code
        micr_match = re.search(r"MICR\s*Code\s*[:]\s*(\d+)", line, re.IGNORECASE)
        if micr_match:
            info.micr_code = micr_match.group(1).strip()
        
        # IFSC Code
        ifsc_match = re.search(r"IFSC\s*Code\s*[:]\s*([A-Z0-9]+)", line, re.IGNORECASE)
        if ifsc_match:
            info.ifsc_code = ifsc_match.group(1).strip()
        
        # Account number
        acct_match = re.search(r"Account\s*(?:Number|No\.?|#)[:\-\s]*([0-9Xx\-\s]{4,})", line, re.IGNORECASE)
        if acct_match:
            info.account_number = acct_match.group(1).strip()
        elif re.search(r"A/c\.?[:\-\s]*([0-9Xx\-\s]{4,})", line, re.IGNORECASE):
            acct_match = re.search(r"A/c\.?[:\-\s]*([0-9Xx\-\s]{4,})", line, re.IGNORECASE)
            info.account_number = acct_match.group(1).strip()
    
    # Name extraction
    name_patterns = [
        r"Name[:\-]\s*(.+)",
        r"(Account Holder|Customer)[:\-]\s*(.+)",
        r"Dear\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)"
    ]
    
    for pattern in name_patterns:
        name_match = re.search(pattern, text, re.IGNORECASE)
        if name_match:
            if len(name_match.groups()) == 1:
                info.name = name_match.group(1).strip()
            else:
                info.name = name_match.group(2).strip()
            break
    
    # Statement period
    period_patterns = [
        r"Statement\s*Period[:\-]\s*([\w\d\s,\-/]+)",
        r"(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\s*(?:to|-)\s*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})"
    ]
    
    for pattern in period_patterns:
        period_match = re.search(pattern, text, re.IGNORECASE)
        if period_match:
            if len(period_match.groups()) == 1:
                info.statement_period = period_match.group(1).strip()
            else:
                info.statement_period = f"{period_match.group(1)} to {period_match.group(2)}"
            break
    
    return info

def normalize_transactions(rows: List[Dict[str, Any]]) -> List[Transaction]:
    """Enhanced transaction normalization with better Kotak format handling"""
    normalized: List[Transaction] = []
    
    print(f"🔄 Normalizing {len(rows)} raw rows...")
    
    for idx, r in enumerate(rows):
        # Convert keys to lowercase for easier matching
        lr = {str(k).lower().replace(' ', '_').replace('/', '_'): v for k, v in r.items() if v is not None}
        
        # Debug first few rows
        if idx < 5:
            print(f"   Row {idx}: {lr}")
        
        tx_date = None
        tx_desc = None
        tx_ref = None
        tx_type = None
        tx_debit = None
        tx_credit = None
        tx_amount = None
        tx_balance = None
        
        # Extract date with better validation
        if 'date' in lr and lr['date']:
            date_str = str(lr['date']).strip()
            # Skip obvious non-dates
            if not any(word in date_str.lower() for word in ['date', 'narration', 'balance', 'withdrawal']):
                tx_date = parse_date(date_str)
        
        # Extract description (narration in Kotak format)
        if 'description' in lr and lr['description']:
            desc_val = str(lr['description']).strip()
            # Skip header-like content
            if desc_val.lower() not in ['narration', 'description', 'particulars', 'details'] and len(desc_val) > 3:
                tx_desc = desc_val
        
        # Extract reference (Chq/Ref No in Kotak format)
        if 'reference' in lr and lr['reference']:
            ref_val = str(lr['reference']).strip()
            # Skip header-like content and preserve full reference
            if ref_val.lower() not in ['chq/ref no', 'reference', 'ref no', 'chq_ref_no'] and len(ref_val) > 2:
                tx_ref = ref_val
        
        # Enhanced amount extraction for Kotak format: "1,200.00(Dr)" or "25.00(Dr)"
        if 'amount' in lr and lr['amount']:
            amount_str = str(lr['amount']).strip()
            # Skip header-like content
            if amount_str.lower() not in ['withdrawal(dr)/deposit(cr)', 'amount', 'debit/credit', 'balance']:
                amount, tx_type_detected = clean_amount_with_brackets(amount_str)
                if amount is not None:
                    tx_amount = amount
                    tx_type = tx_type_detected
                    
                    # Set debit/credit based on type
                    if tx_type_detected == 'Debit':
                        tx_debit = amount
                        tx_credit = None
                    elif tx_type_detected == 'Credit':
                        tx_credit = amount
                        tx_debit = None
        
        # Extract balance with proper Cr/Dr handling
        if 'balance' in lr and lr['balance']:
            balance_str = str(lr['balance']).strip()
            if balance_str.lower() not in ['balance', 'bal']:
                balance_amount, balance_type = clean_amount_with_brackets(balance_str)
                if balance_amount is not None:
                    tx_balance = balance_amount
                    # Note: Balance type (Cr/Dr) indicates account state, not transaction type
        
        # Fallback extraction for other possible column names
        if tx_amount is None:
            # Look for withdrawal/deposit columns
            if 'withdrawal' in lr and lr['withdrawal']:
                withdrawal_str = str(lr['withdrawal']).strip()
                if withdrawal_str.lower() not in ['withdrawal', 'debit', 'dr'] and withdrawal_str != '':
                    amount, _ = clean_amount_with_brackets(withdrawal_str)
                    if amount is not None:
                        tx_amount = amount
                        tx_debit = amount
                        tx_type = 'Debit'
            
            if 'deposit' in lr and lr['deposit']:
                deposit_str = str(lr['deposit']).strip()
                if deposit_str.lower() not in ['deposit', 'credit', 'cr'] and deposit_str != '':
                    amount, _ = clean_amount_with_brackets(deposit_str)
                    if amount is not None:
                        tx_amount = amount
                        tx_credit = amount
                        tx_type = 'Credit'
        
        # Validation: Skip if no meaningful data
        if not tx_date and not tx_desc:
            if idx < 5:  # Debug first few skipped rows
                print(f"   ⚠️ Skipping row {idx}: no date or description")
            continue
        
        # Skip obvious header rows with more specific checks
        if tx_desc and any(word in tx_desc.lower() for word in ['date', 'narration', 'balance', 'withdrawal(dr)/deposit(cr)', 'chq/ref no']):
            print(f"   ⚠️ Skipping header row {idx}: {tx_desc}")
            continue
        
        # Skip rows with invalid dates (like headers)
        if tx_date is None and 'date' in lr:
            date_str = str(lr['date']).lower()
            if any(word in date_str for word in ['date', 'narration', 'balance']):
                print(f"   ⚠️ Skipping invalid date row {idx}: {date_str}")
                continue
        
        # Create transaction
        transaction = Transaction(
            date=tx_date,
            description=tx_desc,
            reference=tx_ref,
            type=tx_type,
            amount=tx_amount,
            debit=tx_debit,
            credit=tx_credit,
            balance=tx_balance,
            raw=r
        )
        
        normalized.append(transaction)
        
        # Debug first few successful transactions
        if len(normalized) <= 5:
            print(f"   ✅ Transaction {len(normalized)}: {tx_date} | {tx_desc[:40] if tx_desc else 'N/A'}... | Ref: {tx_ref} | Amount: {tx_amount} ({tx_type}) | Balance: {tx_balance}")
    
    print(f"🎯 Successfully normalized {len(normalized)} transactions")
    return normalized

def smart_header_detection(table: List[List]) -> Tuple[int, Dict[int, str]]:
    """
    Enhanced header detection that maps table columns to standard field names
    """
    header_row_idx = -1
    column_mapping = {}
    
    # Look for header row in first few rows
    for row_idx, row in enumerate(table[:5]):
        if not row:
            continue
            
        # Check if this row contains header-like content
        header_score = 0
        potential_mapping = {}
        
        for col_idx, cell in enumerate(row):
            if not cell:
                continue
                
            cell_text = str(cell).strip().lower()
            
            # Check against our header mappings
            for field_name, possible_headers in HEADER_MAPPINGS.items():
                for header_variant in possible_headers:
                    if header_variant.lower() in cell_text or cell_text in header_variant.lower():
                        potential_mapping[col_idx] = field_name
                        header_score += 1
                        break
        
        # If we found a good number of matching headers, this is likely our header row
        if header_score >= 3:  # Need at least 3 matching columns
            header_row_idx = row_idx
            column_mapping = potential_mapping
            break
    
    return header_row_idx, column_mapping

def extract_tables_from_pdf(pdf) -> List[Dict[str, Any]]:
    """Enhanced table extraction with better header detection and column mapping"""
    all_rows: List[Dict[str, Any]] = []
    
    for page_num, page in enumerate(pdf.pages):
        print(f"🔍 Processing page {page_num + 1}")
        
        # Strategy 1: Standard table extraction
        tables = []
        try:
            tables = page.extract_tables() or []
            print(f"   Strategy 1: Found {len(tables)} tables")
        except Exception as e:
            print(f"   Strategy 1 failed: {e}")
        
        # Strategy 2: Different table settings for better line detection
        if not tables:
            try:
                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "lines",
                    "horizontal_strategy": "lines",
                    "min_words_vertical": 1,
                    "min_words_horizontal": 1,
                    "snap_tolerance": 3,
                    "join_tolerance": 3
                })
                print(f"   Strategy 2: Found {len(tables)} tables")
            except Exception as e:
                print(f"   Strategy 2 failed: {e}")
        
        # Strategy 3: Text-based extraction for complex layouts
        if not tables:
            try:
                tables = page.extract_tables(table_settings={
                    "vertical_strategy": "text",
                    "horizontal_strategy": "text",
                    "intersection_tolerance": 5,
                    "text_tolerance": 3
                })
                print(f"   Strategy 3: Found {len(tables)} tables")
            except Exception as e:
                print(f"   Strategy 3 failed: {e}")
        
        # Process extracted tables
        for table_idx, table in enumerate(tables):
            if not table or len(table) < 2:
                continue
            
            print(f"   📋 Processing table {table_idx + 1} with {len(table)} rows")
            
            # Debug: print first few rows to understand structure
            for i, row in enumerate(table[:3]):
                print(f"      Row {i}: {[str(cell)[:30] + '...' if cell and len(str(cell)) > 30 else cell for cell in row]}")
            
            # Use smart header detection
            header_row_idx, column_mapping = smart_header_detection(table)
            
            if header_row_idx >= 0:
                print(f"      ✅ Found header at row {header_row_idx}")
                print(f"      📊 Column mapping: {column_mapping}")
                header = table[header_row_idx]
                data_rows = table[header_row_idx + 1:]
            else:
                print(f"      ⚠️ No clear header found, assuming standard format")
                # Fallback: assume standard Kotak format
                column_mapping = {
                    0: 'date',
                    1: 'description', 
                    2: 'reference',
                    3: 'amount',
                    4: 'balance'
                }
                header = table[0] if table else []
                data_rows = table[1:] if len(table) > 1 else table
            
            if not data_rows:
                print(f"      No data rows found")
                continue
            
            # Process data rows with proper column mapping
            processed_count = 0
            for row_idx, row in enumerate(data_rows):
                if not row or all(not cell or str(cell).strip() == '' for cell in row):
                    continue
                
                # Skip rows that look like headers (contain header keywords)
                if any(cell and any(header_word in str(cell).lower() for header_word in ['date', 'narration', 'balance', 'withdrawal', 'deposit']) for cell in row):
                    continue
                
                # Create record with proper field mapping
                record = {
                    "_page": page_num + 1, 
                    "_table": table_idx, 
                    "_row": row_idx,
                    "_extraction_method": "table_with_mapping"
                }
                
                # Map columns to standard field names
                for col_idx, cell in enumerate(row):
                    if cell and str(cell).strip():
                        field_name = column_mapping.get(col_idx, f"col{col_idx}")
                        
                        # Handle special case for Kotak's combined withdrawal/deposit column
                        if field_name == 'amount' and cell:
                            cell_str = str(cell).strip()
                            # Check if it's in format like "1,200.00(Dr)" or "25.00(Dr)"
                            if '(dr)' in cell_str.lower() or '(cr)' in cell_str.lower():
                                record['amount'] = cell_str
                            else:
                                record['amount'] = cell_str
                        else:
                            record[field_name] = str(cell).strip()
                
                # Only add row if it has meaningful transaction data
                if ('date' in record and record['date']) or ('description' in record and record['description']):
                    all_rows.append(record)
                    processed_count += 1
            
            print(f"      ✅ Processed {processed_count} rows from table {table_idx + 1}")
    
    print(f"📊 Total extracted rows: {len(all_rows)}")
    return all_rows

def build_output_structure(
    bank_name: Optional[str], 
    customer_info: CustomerInfo, 
    images: List[str], 
    transactions: List[Transaction], 
    inferred_currency: Optional[str]
) -> Dict[str, Any]:
    return {
        "bank": bank_name or None,
        "customer": asdict(customer_info),
        "currency": inferred_currency,
        "logo_images_base64": images,
        "transactions": [asdict(t) for t in transactions],
        "summary": {
            "total_transactions": len(transactions),
            "total_credits": len([t for t in transactions if t.type == 'Credit']),
            "total_debits": len([t for t in transactions if t.type == 'Debit'])
        }
    }

def process_pdf(path: str, output_path: str, password: Optional[str] = None, use_ocr: bool = False):
    """
    Main processing function that chooses between regular PDF extraction and column-wise OCR
    """
    if use_ocr:
        # Use enhanced column-wise OCR processing
        process_pdf_with_column_ocr(path, output_path, password)
    else:
        # Use original PDF extraction method
        if not os.path.exists(path):
            raise FileNotFoundError(f"Input file not found: {path}")
        
        pdf = open_pdf_with_password(path, password=password)
        
        try:
            # Extract text for metadata
            text_snippet = extract_text_blocks(pdf)
            customer_info = extract_customer_info(text_snippet)
            
            # Extract bank name from first page
            bank_name = None
            if pdf.pages:
                first_page_text = pdf.pages[0].extract_text() or ""
                lines = [line.strip() for line in first_page_text.splitlines() if line.strip()]
                for line in lines[:5]:  # Check first 5 lines
                    if line and len(line) > 5 and not re.match(r'^[\d\s\-/]+$', line):
                        bank_name = line
                        break
            
            # Extract images (logos)
            images_b64 = extract_images_from_page(pdf.pages[0]) if pdf.pages else []
            
            # Extract and normalize transactions
            raw_rows = extract_tables_from_pdf(pdf)
            print(f"Extracted {len(raw_rows)} raw table rows")
            
            transactions = normalize_transactions(raw_rows)
            print(f"Normalized to {len(transactions)} transactions")
            
            # Guess currency
            currency = guess_currency(text_snippet)
            if not currency and pdf.pages:
                first_page_text = pdf.pages[0].extract_text() or ""
                currency = guess_currency(first_page_text)
            
            # Build output structure
            output = build_output_structure(bank_name, customer_info, images_b64, transactions, currency)
            
            # Save to JSON
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output, f, ensure_ascii=False, indent=4)
            
            print(f"✅ Successfully saved JSON to {output_path}")
            print(f"📊 Found {len(transactions)} transactions")
            print(f"🏦 Bank: {bank_name or 'Unknown'}")
            print(f"👤 Customer: {customer_info.name or 'Unknown'}")
            print(f"💳 Account: {customer_info.account_number or 'Unknown'}")
            
        finally:
            try:
                if pdf:
                    pdf.close()
            except Exception:
                pass

def parse_args(argv: List[str]) -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Convert bank statement PDF to JSON (Enhanced Column-wise OCR version)")
    p.add_argument("input", help="Input PDF file")
    p.add_argument("output", help="Output JSON file")
    p.add_argument("--password", help="PDF password (if known)", default=None)
    p.add_argument("--ocr", action="store_true", help="Use enhanced column-wise OCR for scanned PDFs")
    return p.parse_args(argv)

def main_cli(argv: List[str]):
    args = parse_args(argv)
    if not os.path.exists(args.input):
        print(f"❌ Input file not found: {args.input}")
        sys.exit(2)
    
    try:
        process_pdf(args.input, args.output, password=args.password, use_ocr=args.ocr)
    except RuntimeError as e:
        print(f"❌ Error: {e}")
        sys.exit(3)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(4)

if __name__ == "__main__":
    main_cli(sys.argv[1:])