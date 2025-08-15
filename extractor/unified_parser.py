"""
Unified Bank Statement Parser

This module provides a single, intelligent parsing system that automatically
detects the best extraction method for any bank statement PDF:

1. Tabula-based table extraction (for well-structured PDFs)
2. Direct text parsing with regex (for complex/split header PDFs)
3. Hybrid approach (combines both methods for maximum coverage)

The system automatically chooses the optimal approach based on PDF characteristics.
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple, Union
import pandas as pd
from datetime import datetime
from collections import defaultdict
import pdfplumber
from tabula.io import read_pdf

logger = logging.getLogger(__name__)


class UnifiedStatementParser:
    """
    Unified parser that intelligently combines multiple extraction methods
    to provide maximum compatibility across different bank statement formats.
    """
    
    def __init__(self, pdf_path: str, password: Optional[str] = None, debug: bool = False):
        self.pdf_path = pdf_path
        self.password = password
        self.debug = debug
        
        # Bank-specific patterns and configurations
        self.bank_configs = {
            'kotak': {
                'keywords': ['kotak mahindra bank', 'kmbl', 'kotak.com', 'withdrawal(dr)/deposit(cr)', 'narration', 'transaction details', 'debit', 'credit'],
                'amount_patterns': [
                    r'(\d{1,3}(?:,\d{3})*\.?\d*)\s*\((Dr|Cr)\)',  # Primary: Dr/Cr pattern
                    r'(\d{1,3}(?:,\d{3})*\.?\d*)\s*(Dr|Cr)',  # Alternative: Dr/Cr without parentheses
                    r'(\d{1,3}(?:,\d{3})*\.?\d*)',  # Fallback: any amount
                    r'(\d+\.?\d*)',  # Most basic: any number
                ],
                'date_patterns': [
                    r'(\d{2}-\d{2}-\d{4})',  # DD-MM-YYYY
                    r'(\d{2}/\d{2}/\d{4})',  # DD/MM/YYYY
                    r'(\d{2}\s+[A-Za-z]{3},\s+\d{4})',  # DD MMM, YYYY
                    r'(\d{4}-\d{2}-\d{2})',  # YYYY-MM-DD
                    r'(\d{1,2}-\d{1,2}-\d{2,4})',  # Flexible DD-MM-YYYY
                    r'(\d{1,2}/\d{1,2}/\d{2,4})',  # Flexible DD/MM/YYYY
                ],
                'preferred_method': 'hybrid'  # Try both tabula and text
            },
            'hdfc': {
                'keywords': ['hdfc', 'debit', 'credit', 'transaction details'],
                'amount_patterns': [r'(\d{1,3}(?:,\d{3})*\.?\d*)'],
                'date_patterns': [r'(\d{2}/\d{2}/\d{4})', r'(\d{2}-\d{2}-\d{4})'],
                'preferred_method': 'tabula'  # HDFC usually works well with tabula
            },
            'icici': {
                'keywords': ['icici', 'debit', 'credit'],
                'amount_patterns': [r'(\d{1,3}(?:,\d{3})*\.?\d*)'],
                'date_patterns': [r'(\d{2}/\d{2}/\d{4})', r'(\d{2}-\d{2}-\d{4})'],
                'preferred_method': 'tabula'
            },
            'sbi': {
                'keywords': ['sbi', 'state bank', 'debit', 'credit'],
                'amount_patterns': [r'(\d{1,3}(?:,\d{3})*\.?\d*)'],
                'date_patterns': [r'(\d{2}/\d{2}/\d{4})', r'(\d{2}-\d{2}-\d{4})'],
                'preferred_method': 'tabula'
            },
            'generic': {
                'keywords': [],
                'amount_patterns': [r'(\d{1,3}(?:,\d{3})*\.?\d*)', r'(\d{1,3}(?:,\d{3})*\.?\d*)\s*\((Dr|Cr)\)'],
                'date_patterns': [r'(\d{2}/\d{2}/\d{4})', r'(\d{2}-\d{2}-\d{4})', r'(\d{4}-\d{2}-\d{2})'],
                'preferred_method': 'hybrid'
            }
        }
    
    def parse_statement(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Main parsing method that intelligently chooses the best approach.
        
        Returns:
            Tuple of (transactions, parsing_info)
        """
        parsing_info = {
            'total_methods_tried': 0,
            'successful_methods': [],
            'failed_methods': [],
            'bank_detected': None,
            'primary_method': None,
            'fallback_used': False,
            'transactions_found': 0,
            'confidence_score': 0.0
        }
        
        try:
            # Step 1: Extract text for analysis
            if self.debug:
                print("🔍 Step 1: Extracting text for analysis...")
            
            raw_text = self._extract_text_safe()
            if not raw_text:
                parsing_info['failed_methods'].append('text_extraction')
                return [], parsing_info
            
            # Step 2: Detect bank and statement characteristics
            if self.debug:
                print("🔍 Step 2: Detecting bank and statement characteristics...")
                
            bank_info = self._detect_bank_and_format(raw_text)
            parsing_info['bank_detected'] = bank_info['bank']
            
            if self.debug:
                print(f"   Detected bank: {bank_info['bank']}")
                print(f"   Confidence: {bank_info['confidence']:.2f}")
                print(f"   Preferred method: {bank_info['preferred_method']}")
            
            # Step 3: Choose and execute parsing strategy
            if self.debug:
                print("🔍 Step 3: Executing parsing strategy...")
            
            transactions = []
            
            # Get bank configuration
            bank_config = self.bank_configs.get(bank_info['bank'], self.bank_configs['generic'])
            preferred_method = bank_config['preferred_method']
            parsing_info['primary_method'] = preferred_method
            
            # Execute parsing based on preferred method
            if preferred_method == 'tabula':
                transactions = self._parse_with_tabula_first(raw_text, bank_config, parsing_info)
            elif preferred_method == 'text':
                transactions = self._parse_with_text_first(raw_text, bank_config, parsing_info)
            else:  # hybrid
                transactions = self._parse_with_hybrid_approach(raw_text, bank_config, parsing_info)
            
            # Step 4: Enhance and validate results
            if transactions:
                transactions = self._enhance_transactions(transactions, bank_info)
                parsing_info['transactions_found'] = len(transactions)
                parsing_info['confidence_score'] = self._calculate_confidence_score(transactions, bank_info)
                
                if self.debug:
                    print(f"✅ Successfully extracted {len(transactions)} transactions")
                    print(f"   Confidence score: {parsing_info['confidence_score']:.2f}")
            else:
                if self.debug:
                    print("❌ No transactions extracted")
            
            return transactions, parsing_info
            
        except Exception as e:
            if self.debug:
                print(f"❌ Unified parser failed: {e}")
            parsing_info['failed_methods'].append(f'unified_parser_error: {str(e)}')
            return [], parsing_info
    
    def _extract_text_safe(self) -> str:
        """Safely extract text from PDF with proper password handling."""
        try:
            import PyPDF2
            
            with open(self.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                
                # Handle encrypted PDF
                if reader.is_encrypted:
                    if self.password:
                        if not reader.decrypt(self.password):
                            if self.debug:
                                print("❌ Incorrect password provided")
                            return ""
                    else:
                        if not reader.decrypt(''):
                            if self.debug:
                                print("❌ PDF is encrypted and requires password")
                            return ""
                
                # Extract text from all pages
                text = ""
                for page in reader.pages:
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                    except Exception as e:
                        if self.debug:
                            print(f"⚠️ Error extracting from page: {e}")
                        continue
                
                return text
                
        except Exception as e:
            if self.debug:
                print(f"❌ Text extraction failed: {e}")
            return ""
    
    def _detect_bank_and_format(self, text: str) -> Dict[str, Any]:
        """Detect bank and statement format from text."""
        text_lower = text.lower()
        bank_scores = {}
        
        # Score each bank based on keyword matches with weighted scoring
        for bank, config in self.bank_configs.items():
            if bank == 'generic':
                continue
                
            score = 0
            matched_keywords = []
            unique_bank_identifiers = 0
            
            for keyword in config['keywords']:
                if keyword in text_lower:
                    score += 1
                    matched_keywords.append(keyword)
                    
                    # Count unique bank identifiers (not generic terms)
                    if keyword not in ['debit', 'credit', 'transaction details', 'narration']:
                        unique_bank_identifiers += 1
            
            if score > 0:
                # Weighted scoring: prioritize unique bank identifiers
                base_score = score / len(config['keywords'])
                unique_bonus = unique_bank_identifiers * 0.2  # Bonus for unique identifiers
                final_score = min(1.0, base_score + unique_bonus)
                
                bank_scores[bank] = final_score
                if self.debug:
                    print(f"   {bank.upper()}: {score}/{len(config['keywords'])} keywords matched: {matched_keywords}")
                    print(f"   {bank.upper()}: Unique identifiers: {unique_bank_identifiers}")
                    print(f"   {bank.upper()}: Base score = {base_score:.2f}, Bonus = {unique_bonus:.2f}, Final = {final_score:.2f}")
        
        # Determine best match
        if bank_scores:
            best_bank = max(bank_scores, key=bank_scores.get)
            confidence = bank_scores[best_bank]
            if self.debug:
                print(f"   🏆 Best match: {best_bank.upper()} (confidence: {confidence:.2f})")
        else:
            best_bank = 'generic'
            confidence = 0.5  # Default confidence for generic
            if self.debug:
                print(f"   🏆 No bank matches, using generic (confidence: {confidence:.2f})")
        
        # Analyze format characteristics
        format_analysis = self._analyze_format_characteristics(text)
        
        return {
            'bank': best_bank,
            'confidence': confidence,
            'preferred_method': self.bank_configs[best_bank]['preferred_method'],
            'format_analysis': format_analysis
        }
    
    def _analyze_format_characteristics(self, text: str) -> Dict[str, Any]:
        """Analyze text to understand PDF structure and formatting."""
        analysis = {
            'has_clear_table_structure': False,
            'has_split_headers': False,
            'estimated_transaction_count': 0,
            'date_format': None,
            'amount_format': None
        }
        
        lines = text.split('\n')
        
        # Check for table-like structure
        table_indicators = ['date', 'amount', 'balance', 'description', 'debit', 'credit']
        header_lines = [line for line in lines[:10] if any(indicator in line.lower() for indicator in table_indicators)]
        analysis['has_clear_table_structure'] = len(header_lines) > 0
        
        # Check for split headers (like Kotak's "Withdrawal(Dr)/" + "Deposit(Cr)")
        split_patterns = [r'withdrawal\(dr\)/', r'deposit\(cr\)', r'debit.*credit']
        analysis['has_split_headers'] = any(re.search(pattern, text.lower()) for pattern in split_patterns)
        
        # Estimate transaction count
        date_patterns = [r'\d{2}-\d{2}-\d{4}', r'\d{2}/\d{2}/\d{4}', r'\d{4}-\d{2}-\d{2}']
        date_matches = sum(len(re.findall(pattern, text)) for pattern in date_patterns)
        analysis['estimated_transaction_count'] = date_matches
        
        # Detect date format
        if re.search(r'\d{2}-\d{2}-\d{4}', text):
            analysis['date_format'] = 'DD-MM-YYYY'
        elif re.search(r'\d{2}/\d{2}/\d{4}', text):
            analysis['date_format'] = 'DD/MM/YYYY'
        elif re.search(r'\d{4}-\d{2}-\d{2}', text):
            analysis['date_format'] = 'YYYY-MM-DD'
        
        # Detect amount format
        if re.search(r'\d+\.?\d*\s*\(Dr\)', text):
            analysis['amount_format'] = 'Dr/Cr_suffix'
        elif re.search(r'\(\d+\.?\d*\)', text):
            analysis['amount_format'] = 'parentheses_negative'
        else:
            analysis['amount_format'] = 'standard'
        
        return analysis
    
    def _parse_with_tabula_first(self, raw_text: str, bank_config: Dict, parsing_info: Dict) -> List[Dict[str, Any]]:
        """Try tabula first, fallback to text parsing if needed."""
        parsing_info['total_methods_tried'] += 1
        
        try:
            if self.debug:
                print("   🔧 Trying tabula-based extraction...")
                
            # Extract tables using tabula
            tabula_options = {
                "pages": "all",
                "multiple_tables": True,
                "guess": True,
                "lattice": True,
                "stream": True,
                "password": self.password
            }
            
            tables = read_pdf(self.pdf_path, **tabula_options)
            
            if tables and len(tables) > 0:
                # Process tables with bank-specific parser
                transactions = self._process_tabula_tables(tables, bank_config)
                
                if transactions:
                    parsing_info['successful_methods'].append('tabula')
                    if self.debug:
                        print(f"   ✅ Tabula extracted {len(transactions)} transactions")
                    return transactions
            
            parsing_info['failed_methods'].append('tabula')
            if self.debug:
                print("   ❌ Tabula failed, trying text parsing...")
            
            # Fallback to text parsing
            parsing_info['fallback_used'] = True
            return self._parse_with_text_method(raw_text, bank_config, parsing_info)
            
        except Exception as e:
            parsing_info['failed_methods'].append(f'tabula_error: {str(e)}')
            if self.debug:
                print(f"   ❌ Tabula error: {e}, trying text parsing...")
            
            # Fallback to text parsing
            parsing_info['fallback_used'] = True
            return self._parse_with_text_method(raw_text, bank_config, parsing_info)
    
    def _parse_with_text_first(self, raw_text: str, bank_config: Dict, parsing_info: Dict) -> List[Dict[str, Any]]:
        """Try text parsing first, fallback to tabula if needed."""
        parsing_info['total_methods_tried'] += 1
        
        try:
            if self.debug:
                print("   🔧 Trying text-based extraction...")
            
            transactions = self._parse_with_text_method(raw_text, bank_config, parsing_info)
            
            if transactions:
                parsing_info['successful_methods'].append('text_parsing')
                if self.debug:
                    print(f"   ✅ Text parsing extracted {len(transactions)} transactions")
                return transactions
            
            parsing_info['failed_methods'].append('text_parsing')
            if self.debug:
                print("   ❌ Text parsing failed, trying tabula...")
            
            # Fallback to tabula
            parsing_info['fallback_used'] = True
            return self._parse_with_tabula_first(raw_text, bank_config, parsing_info)
            
        except Exception as e:
            parsing_info['failed_methods'].append(f'text_parsing_error: {str(e)}')
            if self.debug:
                print(f"   ❌ Text parsing error: {e}, trying tabula...")
            
            # Fallback to tabula
            parsing_info['fallback_used'] = True
            return self._parse_with_tabula_first(raw_text, bank_config, parsing_info)
    
    def _parse_with_hybrid_approach(self, raw_text: str, bank_config: Dict, parsing_info: Dict) -> List[Dict[str, Any]]:
        """Use hybrid approach - try both methods and combine/choose best results."""
        parsing_info['total_methods_tried'] += 2
        
        if self.debug:
            print("   🔧 Trying hybrid approach (both tabula and text)...")
        
        tabula_transactions = []
        text_transactions = []
        
        # Try tabula
        try:
            tabula_options = {
                "pages": "all",
                "multiple_tables": True,
                "guess": True,
                "lattice": True,
                "stream": True,
                "password": self.password
            }
            
            tables = read_pdf(self.pdf_path, **tabula_options)
            if tables:
                tabula_transactions = self._process_tabula_tables(tables, bank_config)
                if tabula_transactions:
                    parsing_info['successful_methods'].append('tabula')
                    if self.debug:
                        print(f"   ✅ Tabula: {len(tabula_transactions)} transactions")
                else:
                    parsing_info['failed_methods'].append('tabula')
            
        except Exception as e:
            parsing_info['failed_methods'].append(f'tabula_error: {str(e)}')
            if self.debug:
                print(f"   ❌ Tabula failed: {e}")
        
        # Try text parsing
        try:
            text_transactions = self._parse_with_text_method(raw_text, bank_config, parsing_info, track_method=False)
            if text_transactions:
                parsing_info['successful_methods'].append('text_parsing')
                if self.debug:
                    print(f"   ✅ Text parsing: {len(text_transactions)} transactions")
            else:
                parsing_info['failed_methods'].append('text_parsing')
                
        except Exception as e:
            parsing_info['failed_methods'].append(f'text_parsing_error: {str(e)}')
            if self.debug:
                print(f"   ❌ Text parsing failed: {e}")
        
        # Choose best results
        if tabula_transactions and text_transactions:
            # Both worked - choose the one with more transactions or better quality
            if len(text_transactions) > len(tabula_transactions):
                if self.debug:
                    print(f"   📊 Choosing text parsing ({len(text_transactions)} vs {len(tabula_transactions)})")
                return text_transactions
            else:
                if self.debug:
                    print(f"   📊 Choosing tabula ({len(tabula_transactions)} vs {len(text_transactions)})")
                return tabula_transactions
        elif text_transactions:
            if self.debug:
                print("   📊 Using text parsing results")
            return text_transactions
        elif tabula_transactions:
            if self.debug:
                print("   📊 Using tabula results")
            return tabula_transactions
        else:
            if self.debug:
                print("   ❌ Both methods failed")
            return []
    
    def _parse_with_text_method(self, raw_text: str, bank_config: Dict, parsing_info: Dict, track_method: bool = True) -> List[Dict[str, Any]]:
        """Parse transactions directly from text using regex patterns."""
        if track_method:
            parsing_info['total_methods_tried'] += 1
        
        transactions = []
        lines = raw_text.split('\n')
        
        if self.debug:
            print(f"   📄 Processing {len(lines)} lines of text...")
            print(f"   🔍 First 10 lines for debugging:")
            for i, line in enumerate(lines[:10]):
                print(f"      Line {i+1}: '{line[:100]}...'")
        
        # Use bank-specific patterns
        amount_patterns = bank_config['amount_patterns']
        date_patterns = bank_config['date_patterns']
        
        # Enhanced line processing with better filtering
        for line_num, line in enumerate(lines):
            line = line.strip()
            
            # More lenient line filtering for Kotak statements
            if not line:
                continue
            
            # Skip obvious header/footer lines but be less restrictive
            line_lower = line.lower()
            if any(skip in line_lower for skip in ['page', 'opening balance', 'closing balance', 'total']):
                continue
            
            # Try to match transaction pattern
            transaction = self._parse_transaction_line(line, amount_patterns, date_patterns)
            if transaction:
                if self.debug:
                    print(f"   ✅ Line {line_num+1}: Found transaction - {transaction['date']} | {transaction['amount']} | {transaction['description'][:50]}...")
                transactions.append(transaction)
            elif self.debug and len(line) > 30:  # Debug longer lines that might contain transactions
                print(f"   🔍 Line {line_num+1}: No transaction found in: '{line[:80]}...'")
        
        if self.debug:
            print(f"   📊 Total transactions found: {len(transactions)}")
        
        if track_method and transactions:
            parsing_info['successful_methods'].append('text_parsing')
        elif track_method:
            parsing_info['failed_methods'].append('text_parsing')
        
        return transactions
    
    def _parse_transaction_line(self, line: str, amount_patterns: List[str], date_patterns: List[str]) -> Optional[Dict[str, Any]]:
        """Parse a single transaction line using flexible patterns."""
        try:
            # Skip header lines and empty lines
            line_lower = line.lower().strip()
            if (not line or 
                any(header in line_lower for header in ['page', 'opening balance', 'closing balance', 'total']) or
                line_lower.count(' ') < 2):  # More lenient - need at least 2 words
                return None
            
            # Try to find date with more flexible patterns
            date_str = None
            for pattern in date_patterns:
                match = re.search(pattern, line)
                if match:
                    date_str = match.group(1)
                    break
            
            if not date_str:
                return None
            
            # Enhanced amount parsing - look for Dr/Cr indicators first
            transaction_amount = None
            balance = None
            dr_cr_indicator = None
            
            # First, try to find Dr/Cr pattern
            dr_cr_pattern = r'(\d{1,3}(?:,\d{3})*\.?\d*)\s*\((Dr|Cr)\)'
            dr_cr_match = re.search(dr_cr_pattern, line)
            
            if dr_cr_match:
                # Found Dr/Cr pattern - this is likely the transaction amount
                transaction_amount = float(dr_cr_match.group(1).replace(',', ''))
                dr_cr_indicator = dr_cr_match.group(2)
                if dr_cr_indicator == 'Dr':
                    transaction_amount = -transaction_amount
                
                # Look for balance after the Dr/Cr amount
                remaining_line = line[dr_cr_match.end():]
                balance_match = re.search(r'(\d{1,3}(?:,\d{3})*\.?\d*)', remaining_line)
                if balance_match:
                    balance = float(balance_match.group(1).replace(',', ''))
            else:
                # No Dr/Cr pattern - try to find multiple amounts
                amounts = []
                for pattern in amount_patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        try:
                            amount_val = float(match.group(1).replace(',', ''))
                            amounts.append(amount_val)
                        except ValueError:
                            continue
                
                if len(amounts) >= 2:
                    # For Kotak, try to identify which is transaction vs balance
                    # Look for Dr/Cr indicators in the text
                    if 'dr' in line_lower or 'debit' in line_lower:
                        # If debit mentioned, first amount is likely transaction (negative)
                        transaction_amount = -abs(amounts[0])
                        balance = amounts[-1]
                    elif 'cr' in line_lower or 'credit' in line_lower:
                        # If credit mentioned, first amount is likely transaction (positive)
                        transaction_amount = abs(amounts[0])
                        balance = amounts[-1]
                    else:
                        # Default: assume first amount is transaction, last is balance
                        transaction_amount = amounts[0]
                        balance = amounts[-1]
                elif len(amounts) == 1:
                    # Only one amount found - might be transaction amount
                    transaction_amount = amounts[0]
                    balance = None
            
            if transaction_amount is None:
                return None
            
            # Extract description more reliably
            # Find the position after the date
            date_end = line.find(date_str) + len(date_str)
            
            # Find the position of the first amount
            first_amount_pos = len(line)
            for pattern in amount_patterns:
                match = re.search(pattern, line)
                if match:
                    first_amount_pos = min(first_amount_pos, match.start())
            
            # Extract description between date and first amount
            if first_amount_pos > date_end:
                description = line[date_end:first_amount_pos].strip()
            else:
                # Fallback: take text after date, excluding amounts
                description = line[date_end:].strip()
                # Remove amounts from description
                for pattern in amount_patterns:
                    description = re.sub(pattern, '', description)
                description = description.strip()
            
            # Clean up description
            description = re.sub(r'\s+', ' ', description).strip()
            if description.startswith('-') or description.startswith('.'):
                description = description[1:].strip()
            
            # Additional validation and debugging
            if self.debug:
                print(f"🔍 Parsed line: Date='{date_str}' -> '{date}'")
                print(f"🔍 Amount: {transaction_amount} (Dr/Cr: {dr_cr_indicator})")
                print(f"🔍 Balance: {balance}")
                print(f"🔍 Description: '{description}'")
                print(f"🔍 Original line: '{line}'")
                print("---")
            
            # Validate parsed data
            if not description or len(description) < 2:
                if self.debug:
                    print(f"⚠️ Skipping transaction - description too short: '{description}'")
                return None
            
            # Parse date with more formats
            try:
                if '-' in date_str:
                    if len(date_str.split('-')[0]) == 4:  # YYYY-MM-DD
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')
                    else:  # DD-MM-YYYY
                        date_obj = datetime.strptime(date_str, '%d-%m-%Y')
                elif '/' in date_str:
                    if len(date_str.split('/')[0]) == 4:  # YYYY/MM/DD
                        date_obj = datetime.strptime(date_str, '%Y/%m/%d')
                    else:  # DD/MM/YYYY
                        date_obj = datetime.strptime(date_str, '%d/%m/%Y')
                else:
                    # Try common formats
                    for fmt in ['%Y-%m-%d', '%d-%m-%Y', '%Y/%m/%d', '%d/%m/%Y']:
                        try:
                            date_obj = datetime.strptime(date_str, fmt)
                            break
                        except ValueError:
                            continue
                    else:
                        date_obj = datetime.strptime(date_str, '%Y-%m-%d')  # Default
                
                date = date_obj.strftime('%Y-%m-%d')
            except ValueError:
                date = date_str
            
            return {
                'date': date,
                'description': description,
                'ref': '',
                'amount': transaction_amount,
                'balance': balance,
                'category': self._categorize_transaction(description, transaction_amount),
                'parser_confidence': 'high' if dr_cr_indicator else 'medium'
            }
            
        except Exception as e:
            if self.debug:
                print(f"❌ Error parsing line: {e}")
            return None
    
    def _process_tabula_tables(self, tables: List[pd.DataFrame], bank_config: Dict) -> List[Dict[str, Any]]:
        """Process tables extracted by tabula using intelligent column detection."""
        all_transactions = []
        
        for i, df in enumerate(tables):
            if df is None or df.empty:
                continue
            
            if self.debug:
                print(f"   📊 Processing table {i+1}: {df.shape}")
            
            # Clean and analyze table
            df = self._clean_table_columns(df)
            if df is None or df.empty:
                continue
            
            # Use intelligent column detection
            transactions = self._parse_table_intelligent(df, bank_config)
            if transactions:
                all_transactions.extend(transactions)
                if self.debug:
                    print(f"   ✅ Table {i+1}: {len(transactions)} transactions")
        
        return all_transactions
    
    def _parse_table_intelligent(self, df: pd.DataFrame, bank_config: Dict) -> List[Dict[str, Any]]:
        """Parse a table using intelligent column detection."""
        # Identify columns
        col_map = self._identify_columns_smart(df)
        
        if not col_map.get('date') or not (col_map.get('amount') or (col_map.get('debit') and col_map.get('credit'))):
            return []
        
        transactions = []
        
        for _, row in df.iterrows():
            try:
                # Parse date
                date_val = row.get(col_map.get('date'), '')
                date = self._parse_date_flexible(date_val)
                if not date:
                    continue
                
                # Parse description
                desc = row.get(col_map.get('desc'), '') if col_map.get('desc') else ''
                description = str(desc).strip() if not pd.isna(desc) else ''
                
                # Parse amount
                amount = None
                if col_map.get('amount'):
                    amount = self._parse_amount_flexible(row.get(col_map['amount'], ''))
                elif col_map.get('debit') and col_map.get('credit'):
                    debit = self._parse_amount_flexible(row.get(col_map['debit'], ''))
                    credit = self._parse_amount_flexible(row.get(col_map['credit'], ''))
                    if debit is not None and debit != 0:
                        amount = -abs(debit)
                    elif credit is not None and credit != 0:
                        amount = abs(credit)
                
                if amount is None:
                    continue
                
                # Parse balance
                balance = None
                if col_map.get('balance'):
                    balance = self._parse_amount_flexible(row.get(col_map['balance'], ''))
                
                # Parse reference
                ref = row.get(col_map.get('ref', ''), '') if col_map.get('ref') else ''
                reference = str(ref).strip() if not pd.isna(ref) else ''
                
                transaction = {
                    'date': date,
                    'description': description,
                    'ref': reference,
                    'amount': amount,
                    'balance': balance,
                    'category': self._categorize_transaction(description, amount),
                    'parser_confidence': 'high'
                }
                
                transactions.append(transaction)
                
            except Exception as e:
                if self.debug:
                    print(f"❌ Error processing table row: {e}")
                continue
        
        return transactions
    
    def _clean_table_columns(self, df: pd.DataFrame) -> Optional[pd.DataFrame]:
        """Clean DataFrame columns and handle various naming conventions."""
        if df.empty:
            return None
        
        clean_df = df.copy()
        
        # If columns are unnamed/numeric, try using first row as headers
        if all(isinstance(col, (int, float)) or str(col).startswith('Unnamed:') for col in clean_df.columns):
            if not clean_df.empty:
                first_row = [str(val).strip() for val in clean_df.iloc[0]]
                if any(re.search(r'[a-zA-Z]', h) for h in first_row if h):
                    clean_df.columns = first_row
                    clean_df = clean_df.iloc[1:].reset_index(drop=True)
        
        # Clean column names
        new_columns = []
        for col in clean_df.columns:
            if pd.isna(col):
                new_name = f"unnamed_{len(new_columns)}"
            else:
                new_name = str(col).strip().replace('\n', ' ').replace('\r', ' ')
                # Standardize common terms
                new_name = re.sub(r'transaction\s+date', 'Date', new_name, flags=re.IGNORECASE)
                new_name = re.sub(r'narration|particulars|details', 'Description', new_name, flags=re.IGNORECASE)
            new_columns.append(new_name)
        
        clean_df.columns = new_columns
        
        # Remove empty columns
        clean_df.dropna(axis=1, how='all', inplace=True)
        
        return clean_df
    
    def _identify_columns_smart(self, df: pd.DataFrame) -> Dict[str, str]:
        """Intelligently identify column purposes."""
        col_mapping = {}
        cols = list(df.columns)
        
        # Common keywords for each column type
        keywords = {
            'date': ['date', 'txn date', 'transaction date'],
            'desc': ['description', 'details', 'narration', 'particulars', 'memo'],
            'amount': ['amount', 'value', 'withdrawal(dr)/deposit(cr)'],
            'debit': ['debit', 'withdrawal', 'payment', 'outgoing', 'dr'],
            'credit': ['credit', 'deposit', 'payment in', 'incoming', 'cr'],
            'balance': ['balance', 'running balance', 'acct balance'],
            'ref': ['ref', 'reference', 'chq', 'cheque', 'check']
        }
        
        # First pass: match by keywords
        for col_type, terms in keywords.items():
            for col in cols:
                col_lower = str(col).lower()
                if any(term in col_lower for term in terms):
                    col_mapping[col_type] = col
                    break
        
        # Second pass: analyze content if keywords failed
        if 'date' not in col_mapping:
            for col in cols:
                sample = df[col].astype(str).dropna().head(5)
                if sample.empty:
                    continue
                date_pattern = r'(\d{1,2}[/-]\d{1,2}[/-]\d{2,4}|\d{1,2}\s+[A-Za-z]{3})'
                if sum(sample.str.contains(date_pattern, regex=True)) >= 2:
                    col_mapping['date'] = col
                    break
        
        return col_mapping
    
    def _parse_date_flexible(self, date_val: Any) -> Optional[str]:
        """Parse date from various formats."""
        if pd.isna(date_val) or not date_val:
            return None
        
        date_str = str(date_val).strip()
        
        # Try different date formats
        formats = ['%d-%m-%Y', '%d/%m/%Y', '%Y-%m-%d', '%m/%d/%Y', '%d %b %Y', '%d-%b-%Y']
        
        for fmt in formats:
            try:
                date_obj = datetime.strptime(date_str, fmt)
                return date_obj.strftime('%Y-%m-%d')
            except ValueError:
                continue
        
        return None
    
    def _parse_amount_flexible(self, amount_val: Any) -> Optional[float]:
        """Parse amount from various formats."""
        if pd.isna(amount_val) or not amount_val:
            return None
        
        amount_str = str(amount_val).strip()
        
        # Handle Dr/Cr indicators
        is_negative = False
        if amount_str.endswith('(Dr)') or amount_str.endswith(' Dr'):
            is_negative = True
            amount_str = re.sub(r'\s*\(Dr\)|\s*Dr$', '', amount_str)
        elif amount_str.endswith('(Cr)') or amount_str.endswith(' Cr'):
            amount_str = re.sub(r'\s*\(Cr\)|\s*Cr$', '', amount_str)
        
        # Handle parentheses for negatives
        if amount_str.startswith('(') and amount_str.endswith(')'):
            is_negative = True
            amount_str = amount_str[1:-1]
        
        # Remove currency symbols and commas
        amount_str = re.sub(r'[$€£,\s]', '', amount_str)
        
        try:
            amount = float(amount_str)
            return -amount if is_negative else amount
        except ValueError:
            return None
    
    def _categorize_transaction(self, description: str, amount: float) -> str:
        """Categorize transaction based on description and amount."""
        desc_lower = description.lower()
        
        # NACH transactions
        if 'nach' in desc_lower:
            if any(k in desc_lower for k in ['loan', 'emi', 'instalment', 'installment']):
                return 'EMI'
            if any(k in desc_lower for k in ['sip', 'mutual fund', 'investment', 'insurance']):
                return 'Investment'
            return 'NACH'
        
        # Payment methods
        if any(k in desc_lower for k in ['upi', 'gpay', 'google pay', 'phonepe', 'paytm']):
            return 'UPI'
        if 'neft' in desc_lower:
            return 'NEFT'
        if 'imps' in desc_lower:
            return 'IMPS'
        if any(k in desc_lower for k in ['atm', 'cash withdrawal']):
            return 'ATM'
        if any(k in desc_lower for k in ['cheque', 'chq', 'check']):
            return 'Cheque'
        if any(k in desc_lower for k in ['card', 'pos']):
            return 'Card'
        
        # Transaction types
        if any(k in desc_lower for k in ['salary', 'payroll', 'wages']):
            return 'Salary'
        if any(k in desc_lower for k in ['emi', 'loan', 'instalment']):
            return 'EMI'
        if any(k in desc_lower for k in ['interest', 'int.']):
            return 'Interest'
        if any(k in desc_lower for k in ['fee', 'charge', 'penalty']):
            return 'Fee'
        if any(k in desc_lower for k in ['refund', 'reversal']):
            return 'Refund'
        
        return 'Other'
    
    def _enhance_transactions(self, transactions: List[Dict[str, Any]], bank_info: Dict) -> List[Dict[str, Any]]:
        """Enhance transactions with additional metadata."""
        for transaction in transactions:
            transaction['bank_type'] = bank_info['bank'].upper()
            transaction['extraction_method'] = 'unified_parser'
            
            # Ensure required fields exist
            if 'ref' not in transaction:
                transaction['ref'] = ''
            if 'category' not in transaction:
                transaction['category'] = 'Other'
        
        # Sort by date
        transactions.sort(key=lambda x: x.get('date', ''))
        
        return transactions
    
    def _calculate_confidence_score(self, transactions: List[Dict[str, Any]], bank_info: Dict) -> float:
        """Calculate confidence score for the parsing results."""
        if not transactions:
            return 0.0
        
        score = bank_info['confidence']  # Start with bank detection confidence
        
        # Adjust based on data quality
        valid_dates = sum(1 for t in transactions if t.get('date') and re.match(r'\d{4}-\d{2}-\d{2}', t['date']))
        valid_amounts = sum(1 for t in transactions if t.get('amount') is not None)
        valid_descriptions = sum(1 for t in transactions if t.get('description') and len(t['description']) > 5)
        
        data_quality = (valid_dates + valid_amounts + valid_descriptions) / (len(transactions) * 3)
        score = (score + data_quality) / 2
        
        return min(score, 1.0)


# Convenience function for easy integration
def parse_statement_unified(pdf_path: str, password: Optional[str] = None, debug: bool = False) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Convenience function to parse any bank statement using the unified parser.
    
    Args:
        pdf_path: Path to the PDF file
        password: Password for encrypted PDFs
        debug: Enable debug output
        
    Returns:
        Tuple of (transactions, parsing_info)
    """
    parser = UnifiedStatementParser(pdf_path, password, debug)
    return parser.parse_statement()
