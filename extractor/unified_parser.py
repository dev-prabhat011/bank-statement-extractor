"""

Unified Bank Statement Parser - Complete Version

This module provides intelligent parsing that automatically detects the specific
layout of each bank statement and applies the most suitable parsing strategy.

Includes all original methods and parsers for maximum compatibility.
"""

import re
import logging

from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

logger = logging.getLogger(__name__)


class EnhancedBalanceDetector:
    """Enhanced balance detection for bank statements."""
    
    def __init__(self, debug=False):
        self.debug = debug

    _NUM = re.compile(r'[+−-]?\s*\d[\d,]*(?:\.\d{1,2})?')


    def _extract_balance_from_last_column(self, line: str) -> Optional[float]:
        """
        Extracts the balance from the rightmost number in the line.
        This avoids mixing it with debit/credit amounts.
        """
        matches = re.findall(r'(\d[\d,]*\.\d{2})', line)
        if not matches:
            return None

        # Always take the last number → Balance
        balance_str = matches[-1]
        try:
            return float(balance_str.replace(',', ''))
        except ValueError:
            return None


    def extract_balance_improved(self, line: str, transaction_amount: float) -> Optional[float]:
        """Improved balance extraction with multiple strategies."""
        
        # Strategy 1: Look for amounts with Cr/Dr indicators (Kotak format)
        balance = self._extract_cr_dr_balance(line, transaction_amount)
        if balance is not None:
            return balance
        
        # Strategy 2: Look for the last reasonable amount different from transaction
        balance = self._extract_last_reasonable_amount(line, transaction_amount)
        if balance is not None:
            return balance
        
        # Strategy 3: Look for amounts after transaction amount position
        balance = self._extract_positional_balance(line, transaction_amount)
        if balance is not None:
            return balance
        
        return None
    
    def _extract_cr_dr_balance(self, line: str, transaction_amount: float) -> Optional[float]:
        """Extract balance from Cr/Dr format like '64,545.98(Cr)'."""
        # Find all amounts with Cr/Dr indicators
        pattern = r'(\d[\d,]*(?:\.\d{2})?)\s*\(\s*(Cr|Dr)\s*\)'
        matches = list(re.finditer(pattern, line, re.IGNORECASE))
        
        if len(matches) >= 2:
            # If we have transaction amount and balance, balance is usually the last Cr amount
            for match in reversed(matches):
                try:
                    amount = float(match.group(1).replace(',', ''))
                    indicator = match.group(2).upper()
                    
                    # Skip if this is the transaction amount
                    if abs(amount - abs(transaction_amount)) < 0.01:
                        continue
                    
                    # Balance is typically shown as Cr (credit/positive)
                    if indicator == 'CR' and self._is_reasonable_balance(amount):
                        return amount
                        
                except ValueError:
                    continue
        
        return None
    
    def _extract_last_reasonable_amount(self, line: str, transaction_amount: float) -> Optional[float]:
        """Extract the last reasonable amount that's not the transaction amount."""
        # Find all amounts in the line
        amounts_pattern = r'(\d[\d,]*(?:\.\d{1,2})?)'
        matches = list(re.finditer(amounts_pattern, line))
        
        # Process matches from right to left (last to first)
        for match in reversed(matches):
            try:
                amount = float(match.group(1).replace(',', ''))
                
                # Skip if this is the transaction amount
                if abs(amount - abs(transaction_amount)) < 0.01:
                    continue
                
                # Skip unreasonably long numbers (UPI refs, phone numbers)
                if len(match.group(1).replace(',', '').replace('.', '')) > 10:
                    continue
                
                # Check if this could be a balance
                if self._is_reasonable_balance(amount):
                    return amount
                    
            except ValueError:
                continue
        
        return None
    
    def _extract_positional_balance(self, line: str, transaction_amount: float) -> Optional[float]:
        """Extract balance that appears after transaction amount in the line."""
        # Find transaction amount position first
        trans_patterns = [
            rf'[+\-]\s*{re.escape(f"{abs(transaction_amount):.2f}")}',
            rf'[+\-]\s*{re.escape(f"{abs(transaction_amount):,.2f}")}',
            rf'[+\-]\s*{int(abs(transaction_amount))}',
        ]
        
        transaction_end_pos = -1
        for pattern in trans_patterns:
            match = re.search(pattern, line)
            if match:
                transaction_end_pos = match.end()
                break
        
        if transaction_end_pos == -1:
            return None
        
        # Look for amounts after transaction position
        remaining_line = line[transaction_end_pos:]
        amount_matches = re.finditer(r'(\d{1,3}(?:,\d{3})*(?:\.\d{2})?)', remaining_line)
        
        for match in amount_matches:
            try:
                amount = float(match.group(1).replace(',', ''))
                if self._is_reasonable_balance(amount):
                    return amount
            except ValueError:
                continue
        
        return None
    
    def _is_reasonable_balance(self, amount: float) -> bool:
        """Check if amount is reasonable for a bank balance."""
        # More intelligent validation for bank balances
        # Skip unreasonably large amounts (more than 1 crore)
        if amount > 10000000:
            return False
        
        # Balance should be in a reasonable range for a typical bank account
        # Allow small amounts as they could be valid balances
        return 0 < amount <= 10000000  # 0 to 1 crore


class LayoutAwareParser:

    """Layout-aware parser that detects statement structure and applies the most suitable parsing strategy."""
    

    def __init__(self, pdf_path: str, password: Optional[str] = None, debug: bool = False, user_selected_bank: Optional[str] = None):
        self.pdf_path = pdf_path
        self.password = password
        self.debug = debug

        self.user_selected_bank = user_selected_bank  # Store user's bank selection
        
        # Initialize enhanced balance detector
        self.balance_detector = EnhancedBalanceDetector(debug=debug)
        

        # Complete layout detection patterns - restored from original
        self.layout_patterns = {
            'kotak': {

                'single_month': {
                    'name': 'Single Month Format (Credit/Debit)',

                    'indicators': [r'credit.*debit', r'\+.*\-', r'transaction_details.*balance'],
                    'parser': 'single_month_parser'
                },

                'real_6month': {
                    'name': 'Real 6-Month Format (Dr/Cr)',
                    'indicators': [r'withdrawal\(dr\)', r'deposit\(cr\)', r'balance.*\(cr\)'],
                    'parser': 'real_6month_parser'
                },
                'separate_fields': {
                    'name': 'Separate Fields Format (6-Month)',

                    'indicators': [r'withdrawal.*deposit', r'narration.*balance', r'cheque_reference_no'],
                    'parser': 'separate_fields_parser'
                },

                'table': {
                    'name': 'Traditional Table Format',

                    'indicators': [r'date\s+transaction\s+details', r'debit\s+credit\s+balance'],
                    'parser': 'table_parser'
                },

                'consolidated': {
                    'name': 'Consolidated Statement Format',
                    'indicators': [

                        r'consolidated.*statement', 
                        r'period.*statement', 
                        r'monthly.*summary',
                        r'start_date.*end_date',  # Period format like "01-01-2025" to "19-07-2025"
                        r'withdrawal.*deposit',   # Separate withdrawal/deposit fields
                        r'cheque_reference_no',   # Cheque reference number field
                        r'narration.*balance'     # Narration and balance structure
                    ],
                    'parser': 'consolidated_parser'
                },
                'detailed': {
                    'name': 'Detailed Transaction Format',
                    'indicators': [r'detailed.*transactions', r'full.*statement', r'complete.*details'],
                    'parser': 'detailed_parser'
                },
                'summary': {
                    'name': 'Summary Statement Format',
                    'indicators': [r'summary.*statement', r'overview.*transactions', r'brief.*summary'],
                    'parser': 'summary_parser'
                }
            },
            'hdfc': {
                'standard': {
                    'name': 'HDFC Standard Format',
                    'indicators': [r'hdfc.*bank', r'standard.*format', r'regular.*statement'],
                    'parser': 'hdfc_standard_parser'
                },
                'separate_cols': {
                    'name': 'HDFC Separate Columns',
                    'indicators': [r'separate.*columns', r'debit.*credit.*separate', r'withdrawal.*deposit'],
                    'parser': 'hdfc_separate_parser'
                }
            },
            'icici': {
                'standard': {
                    'name': 'ICICI Standard Format',
                    'indicators': [r'icici.*bank', r'standard.*icici', r'regular.*icici'],
                    'parser': 'icici_standard_parser'
                },
                'detailed': {
                    'name': 'ICICI Detailed Format',
                    'indicators': [r'detailed.*icici', r'full.*icici', r'complete.*icici'],
                    'parser': 'icici_detailed_parser'
                }
            },
            'sbi': {
                'standard': {
                    'name': 'SBI Standard Format',
                    'indicators': [r'state.*bank.*india', r'sbi.*bank', r'standard.*sbi'],
                    'parser': 'sbi_standard_parser'
                },
                'separate_cols': {
                    'name': 'SBI Separate Columns',
                    'indicators': [r'separate.*sbi', r'debit.*credit.*sbi', r'withdrawal.*deposit.*sbi'],
                    'parser': 'sbi_separate_parser'
                }
            }
        }
    
    def parse_statement(self, raw_text: str) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:

        """Main parsing method that detects layout and applies appropriate parser."""
        parsing_info = {
            'layout_detected': None,
            'layout_name': None,
            'parser_used': None,
            'transactions_found': 0,
            'confidence_score': 0.0
        }
        
        try:

            # Use user's bank selection if provided, otherwise auto-detect
            if self.user_selected_bank:
                # Normalize user's bank selection to match layout pattern keys
                bank_type = self._normalize_bank_name(self.user_selected_bank)
                if self.debug:
                    print(f"?? Using user-selected bank: {self.user_selected_bank} -> normalized to: {bank_type}")
            else:
                bank_type = self._detect_bank(raw_text)
                if self.debug:
                    print(f"🔍 Auto-detected bank: {bank_type}")
            

            # Detect layout
            layout_info = self._detect_layout(raw_text, bank_type)

            parsing_info.update(layout_info)
            
            if self.debug:
                print(f"?? Detected layout: {layout_info['layout_name']}")
                print(f"?? Using parser: {layout_info['parser']}")
            

            # Apply parser
            transactions = self._apply_layout_parser(raw_text, layout_info)
            
            if transactions:
                # Apply balance validation and fixes
                transactions = self._validate_and_fix_balances(transactions)
                transactions = self._enhance_transactions(transactions, bank_type)
                parsing_info['transactions_found'] = len(transactions)
                parsing_info['confidence_score'] = self._calculate_confidence(transactions, layout_info)
                
                if self.debug:
                    print(f"? Successfully extracted {len(transactions)} transactions")
                    print(f"?? IMPORTANT: All transactions captured for fraud detection analysis")
            
            return transactions, parsing_info
            
        except Exception as e:
            if self.debug:

                print(f"? Parser failed: {e}")
            return [], parsing_info
    
    def _detect_bank(self, text: str) -> str:
        """Detect bank type from text content."""
        text_lower = text.lower()
        

        # Enhanced Kotak detection - look for more patterns
        kotak_indicators = [
            'kotak mahindra bank', 'kmbl', 'kotak.com', 'kotak mahindra',
            'kotak bank', 'kotak mahindra bank limited', 'kotak'
        ]
        if any(keyword in text_lower for keyword in kotak_indicators):
            return 'kotak'

        
        # Check for other banks
        elif any(keyword in text_lower for keyword in ['hdfc bank', 'hdfc.com']):
            return 'hdfc'
        elif any(keyword in text_lower for keyword in ['icici bank', 'icici.com']):
            return 'icici'

        elif any(keyword in text_lower for keyword in ['state bank of india', 'sbi', 'sbi.co.in']):
            return 'sbi'

        
        # If no specific bank detected, check for Kotak-specific patterns in transactions
        # Look for common Kotak transaction patterns
        if any(pattern in text_lower for pattern in ['nach - mut - dr', 'withdrawal', 'deposit', 'cheque_reference_no']):
            if 'kotak' in text_lower or 'kmbl' in text_lower:
                return 'kotak'
        
            return 'generic'
    

    def _normalize_bank_name(self, bank_name: str) -> str:
        """Normalize user's bank selection to match layout pattern keys."""
        bank_lower = bank_name.lower()
        
        # Map common bank names to layout pattern keys
        if any(keyword in bank_lower for keyword in ['kotak', 'kmbl']):
            return 'kotak'
        elif any(keyword in bank_lower for keyword in ['hdfc']):
            return 'hdfc'
        elif any(keyword in bank_lower for keyword in ['icici']):
            return 'icici'
        elif any(keyword in bank_lower for keyword in ['sbi', 'state bank of india']):
            return 'sbi'
        else:
            # If no match found, return the first word (e.g., "Kotak Mahindra Bank" -> "kotak")
            return bank_lower.split()[0]
    
    def _detect_layout(self, text: str, bank_type: str) -> Dict[str, Any]:
        """Detect the specific layout of the statement."""
        if bank_type not in self.layout_patterns:

            return {'layout_type': 'generic', 'layout_name': 'Generic', 'parser': 'generic_parser'}
        
        bank_patterns = self.layout_patterns[bank_type]
        best_match = None
        best_score = 0
        
        for layout_id, layout_info in bank_patterns.items():

            score = sum(1 for indicator in layout_info['indicators'] if re.search(indicator, text, re.IGNORECASE))
            if score > best_score:
                best_score = score
                best_match = {
                    'layout_type': layout_id,
                    'layout_name': layout_info['name'],
                    'parser': layout_info['parser'],

                    'score': score
                }
        
        return best_match or {'layout_type': 'generic', 'layout_name': 'Generic', 'parser': 'generic_parser'}
    
    def _apply_layout_parser(self, raw_text: str, layout_info: Dict) -> List[Dict[str, Any]]:
        """Apply the appropriate parser based on detected layout."""
        parser_method = getattr(self, f"_{layout_info['parser']}", None)
        if parser_method:
            if self.debug:
                print(f"?? Applying {layout_info['parser']}...")

            return parser_method(raw_text)
        return self._generic_parser(raw_text)
    
    def _single_month_parser(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse single month statements with separate credit/debit fields (like May Kotak)."""
        if self.debug:

            print("   ?? Using single month parser...")
        
        transactions = []
        lines = raw_text.split('\n')
        
        if self.debug:
            print(f"   📄 Total lines in text: {len(lines)}")
        
        # Find where transaction data starts (skip headers)
        start_line = self._find_transaction_start(lines)
        if self.debug:
            print(f"   📄 Transaction data starts at line {start_line + 1}")
        
        # Filter lines that look like transactions (starting from transaction data)
        transaction_lines = []
        for line_num, line in enumerate(lines[start_line:], start_line):
            if self._looks_like_transaction_line(line):
                transaction_lines.append((line_num, line))
        
        if self.debug:
            print(f"   📄 Found {len(transaction_lines)} potential transaction lines")
        
        # Combine multi-line transactions
        combined_lines = self._combine_multi_line_transactions([line for _, line in transaction_lines])
        
        if self.debug:
            print(f"   📄 Combined into {len(combined_lines)} transaction lines")
        
        # Parse each combined transaction line
        for line_num, line in combined_lines:
            transaction = self._parse_single_month_line(line)
            if transaction:
                if self.debug:

                    print(f"   ? Line {line_num+1}: Parsed transaction - {transaction['date']} | {transaction['amount']}")
                transactions.append(transaction)
        

        if self.debug:
            print(f"   ?? Single month parser found {len(transactions)} transactions")
            print(f"   ?? Multi-line transactions properly combined for complete descriptions")
        
        return transactions
    
    def _find_transaction_start(self, lines: List[str]) -> int:
        """Find the line number where actual transaction data starts (after column headers)."""
        for i, line in enumerate(lines):
            line_lower = line.lower().strip()
            
            # Skip empty lines
            if not line_lower:
                continue
            
            # Look for column headers that indicate the start of transaction data
            header_patterns = [
                'date',
                'transaction details',
                'debit',
                'credit',
                'balance',
                'narration',
                'withdrawal',
                'deposit'
            ]
            
            # If we find a line that looks like column headers, the next line should be data
            if any(pattern in line_lower for pattern in header_patterns):
                if self.debug:
                    print(f"   📄 Found column headers at line {i+1}: {line[:80]}...")
                return i + 1  # Start from the next line
            
            # Alternative: Look for the first line that has a date pattern
            if re.search(r'\d{1,2}[-/\s][A-Za-z]{3,4}[-/\s]\d{4}', line) or re.search(r'\d{1,2}\s+[A-Za-z]+\s*,\s*\d{4}', line):
                if self.debug:
                    print(f"   📄 Found first date at line {i+1}: {line[:80]}...")
                return i
        
        # If no headers found, start from the beginning
        return 0
    

    def _real_6month_parser(self, raw_text: str) -> List[Dict[str, Any]]:
        """
        Parser for Kotak 6-month consolidated statements.
        Handles multi-line narrations and ensures only complete rows become transactions.
        """
        if self.debug:
            print("   📑 Using improved real 6-month parser...")

        transactions = []
        buffer = ""   # holds partial narration lines

        lines = raw_text.split("\n")
        for idx, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue

            # Skip headers / table titles
            if re.search(r'date\s+narration\s+chq/ref|withdrawal|deposit|balance', line, re.IGNORECASE):
                continue
            if any(kw in line.lower() for kw in ["closing balance", "opening balance", "total"]):
                continue

            # Append line to buffer
            if buffer:
                buffer = buffer + " " + line
            else:
                buffer = line

            # Check if buffer contains BOTH a Dr/Cr txn amount AND a balance
            if re.search(r'\(\s*(Dr|Cr)\s*\)', buffer, re.IGNORECASE) and re.search(r'\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?', buffer):
                txn = self._parse_real_6month_line(buffer)
                if txn:
                    transactions.append(txn)
                    if self.debug:
                        print(f"   ✅ Parsed txn from buffer: {txn}")
                    buffer = ""  # reset after successful parse
                else:
                    if self.debug:
                        print(f"   ⚠️ Could not parse buffer line: {buffer}")

        # If anything left in buffer at the end, try parse once
        if buffer:
            txn = self._parse_real_6month_line(buffer)
            if txn:
                transactions.append(txn)
                if self.debug:
                    print(f"   ✅ Parsed final txn from buffer: {txn}")

        return transactions

    

    def _separate_fields_parser(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse statements with separate withdrawal/deposit fields (like 6-month Kotak)."""
        if self.debug:
            print("   ?? Using separate fields parser...")
        
        transactions = []
        lines = raw_text.split('\n')
        

        # Filter lines that look like transactions
        transaction_lines = []
        for line_num, line in enumerate(lines):

            if self._looks_like_transaction_line(line):
                transaction_lines.append((line_num, line))
        
        # Look for transaction lines with separate fields structure
        for line_num, line in transaction_lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip header lines
            line_lower = line.lower()
            if any(skip in line_lower for skip in ['page', 'opening balance', 'closing balance', 'total', 'date transaction']):
                continue
            
            # Parse separate fields format
            transaction = self._parse_separate_fields_line(line)
            if transaction:
                if self.debug:
                    print(f"   ? Line {line_num+1}: Found transaction - {transaction['date']} | {transaction['amount']}")
                transactions.append(transaction)
        
        if self.debug:
            print(f"   ?? Separate fields parser found {len(transactions)} transactions")
        
        return transactions
    

    def _consolidated_parser(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse consolidated statements with multiple month summaries."""
        if self.debug:

            print("   ?? Using consolidated parser...")
        
        # This parser handles consolidated statements with multiple months
        # It's a specialized version that can handle complex multi-month data
        return self._real_6month_parser(raw_text)  # Use the robust 6-month parser
    
    def _detailed_parser(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse detailed transaction statements with full information."""
        if self.debug:
            print("   ?? Using detailed parser...")
        
        # This parser handles detailed statements with full transaction information
        # It's designed for maximum transaction capture
        return self._single_month_parser(raw_text)  # Use the comprehensive single month parser
    
    def _summary_parser(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse summary statements with overview information."""
        if self.debug:
            print("   📄 Using summary parser...")
        
        # This parser handles summary statements
        # It's optimized for quick overview parsing
        return self._generic_parser(raw_text)  # Use the generic parser for flexibility
    
    def _hdfc_standard_parser(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse HDFC standard format statements."""
        if self.debug:

            print("   ?? Using HDFC standard parser...")
        
        # HDFC standard format parsing logic
        return self._single_month_parser(raw_text)  # Use single month parser as base
    
    def _hdfc_separate_parser(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse HDFC separate columns format statements."""
        if self.debug:

            print("   ?? Using HDFC separate columns parser...")
        

        # HDFC separate columns format parsing logic
        return self._separate_fields_parser(raw_text)  # Use separate fields parser as base
    

    def _icici_standard_parser(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse ICICI standard format statements."""
        if self.debug:

            print("   ?? Using ICICI standard parser...")
        
        # ICICI standard format parsing logic
        return self._single_month_parser(raw_text)  # Use single month parser as base
    
    def _icici_detailed_parser(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse ICICI detailed format statements."""
        if self.debug:
            print("   ?? Using ICICI detailed parser...")
        
        # ICICI detailed format parsing logic
        return self._detailed_parser(raw_text)  # Use detailed parser as base
    
    def _sbi_standard_parser(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse SBI standard format statements."""
        if self.debug:

            print("   ?? Using SBI standard parser...")
        
        # SBI standard format parsing logic
        return self._single_month_parser(raw_text)  # Use single month parser as base
    
    def _sbi_separate_parser(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse SBI separate columns format statements."""
        if self.debug:
            print("   📄 Using SBI separate columns parser...")
        
        # SBI separate columns format parsing logic
        return self._separate_fields_parser(raw_text)  # Use separate fields parser as base
    
    def _looks_like_transaction_line(self, line: str) -> bool:
        """Determine if a line looks like it contains transaction data."""
        if not line or len(line.strip()) < 5:
            return False
        
        line = line.strip()
        line_lower = line.lower()
        
        # DEBUG: Check opening balance specifically
        if 'opening balance' in line_lower:
            if self.debug:
                print(f"   🔍 Found opening balance line: '{line[:80]}...'")
        
        # Skip table headers and non-transactional entries
        skip_patterns = [
            # Table headers
            'date',
            'transaction details',
            'cheque/reference#',
            'debit',
            'credit',
            'narration',
            'withdrawal',
            'deposit',
            'chq/ref',
            
            # Summary rows
            'total debited',
            'total credited', 
            'total transactions',
            'summary',
            'closing balance',
            'statement period',
            'page',
            'generated on',
            'statement generated',
            'computer generated',
            
            # Account info
            'overdraft drawing power',
            'others',
            'sweep td broken',
            'portfolio summary',
            'assets',
            'deposit accounts',
            'savings account',
            'rbi has advised',
            'effective',
            'need help',
            'visit your nearest',
            'call kotak',
            'any discrepancy',
            'ref.no',
            'home branch',
            'crn',
            'ifsc',
            'micr',
            'currency',
            'variant'
        ]
        
        # SPECIAL HANDLING: Don't skip opening balance lines
        if 'opening balance' in line_lower:
            # Skip the normal skip pattern check for opening balance
            pass
        elif any(pattern in line_lower for pattern in skip_patterns):
            if self.debug:

                print(f"   ?? Line filtered out by skip pattern: '{line[:80]}...'")
            return False
        
        # Check for common transaction indicators - STRICT VALIDATION
        has_date = bool(re.search(r'\d{1,2}[-/\s][A-Za-z]{3,4}[-/\s]\d{4}', line) or 
                       re.search(r'\d{1,2}\s+[A-Za-z]+\s*,\s*\d{4}', line))
        
        has_amount = bool(re.search(r'[+-]?\d{1,3}(?:,\d{3})*\.?\d*', line))
        
        has_credit_debit = bool(re.search(r'[+-]\d', line))  # Must have + or - before amount
        
        # Check for meaningful description (more flexible validation)
        has_description = len(line.strip()) > 20  # Reduced from 25 to be more lenient
        
        # Additional check: Description should not be just generic words
        if has_description:
            # Extract description part (after date, before amounts)
            date_match = re.search(r'\d{1,2}\s+[A-Za-z]+\s*,\s*\d{4}', line)
            if date_match:
                date_end = date_match.end()
                # Look for first amount to find description boundaries
                amount_match = re.search(r'[+-]\d', line)
                if amount_match:
                    description_part = line[date_end:amount_match.start()].strip()
                    # Check if description is just generic words
                    generic_words = ['other', 'misc', 'miscellaneous', 'various', 'general', 'unknown', 'na', 'n/a']
                    if description_part.lower() in generic_words:
                        has_description = False
                        if self.debug:
                            print(f"   🚫 Line rejected: Generic description '{description_part}'")
        
        # DEBUG: Show what we found
        if self.debug and 'opening balance' in line_lower:
            print(f"   ?? Opening balance line analysis:")
            print(f"      Has date: {has_date}")
            print(f"      Has amount: {has_amount}")
            print(f"      Has credit/debit: {has_credit_debit}")
            print(f"      Has description: {has_description}")
            print(f"      Line length: {len(line)}")
        
        # FLEXIBLE VALIDATION: Accept lines with date + amount + credit/debit, OR transaction keywords
        if has_date and has_amount and has_credit_debit:
            # Skip lines that are clearly non-transactional (headers, footers, etc.)
            non_transaction_words = ['page', 'total', 'summary', 'statement', 'generated', 'computer']
            if any(word in line_lower for word in non_transaction_words):
                if self.debug:
                    print(f"   🚫 Line rejected: Contains non-transaction words: '{line[:80]}...'")
                return False
            
            # Accept the line if it has date + amount + credit/debit
            if self.debug and 'opening balance' in line_lower:
                print(f"   ? Opening balance line accepted (has date + amount + credit/debit)")
            return True
            # Skip lines that are clearly non-transactional (headers, footers, etc.)
            non_transaction_words = ['page', 'total', 'summary', 'statement', 'generated', 'computer']
            if any(word in line_lower for word in non_transaction_words):
                if self.debug:
                    print(f"   🚫 Line rejected: Contains non-transaction words: '{line[:80]}...'")
                return False
            
            # Accept the line if it has both date and amount
            if self.debug and 'opening balance' in line_lower:
                print(f"   ? Opening balance line accepted (has date + amount)")
            return True
        
        # Check for transaction-specific keywords
        transaction_keywords = ['nach', 'neft', 'upi', 'atm', 'cheque', 'withdrawal', 'deposit', 'transfer']
        if any(keyword in line.lower() for keyword in transaction_keywords):
            if self.debug:
                print(f"   ? Line accepted: Contains transaction keywords: '{line[:80]}...'")
            return True
        
        if self.debug and 'opening balance' in line_lower:
            print(f"   ? Opening balance line rejected")
        
        return False
    
    def _combine_multi_line_transactions(self, lines: List[str]) -> List[Tuple[int, str]]:
        """Intelligently combine multi-line transactions."""
        combined_lines = []
        current_transaction = ""
        current_line_num = 0
        in_transaction = False
        
        for line_num, line in enumerate(lines):
            line = line.strip()
            if not line:
                continue
            
            # Check if this line starts a new transaction (has a date)
            has_date = bool(re.search(r'\d{1,2}\s+[A-Za-z]+\s*,\s*\d{4}', line) or
                           re.search(r'\d{1,2}[-/\s][A-Za-z]{3,4}[-/\s]\d{4}', line))
            has_amounts = bool(re.search(r'[+-]\d', line))
            
            if has_date:
                # New transaction line
                if current_transaction and in_transaction:
                    combined_lines.append((current_line_num, current_transaction))
                    if self.debug:
                        print(f"   ?? Combined transaction {current_line_num+1}: {current_transaction[:80]}...")
                
                current_transaction = line
                current_line_num = line_num
                in_transaction = True
                
            elif has_amounts and in_transaction:
                # Amount line - complete the transaction
                current_transaction += " " + line
                combined_lines.append((current_line_num, current_transaction))
                if self.debug:
                    print(f"   ? Completed transaction {current_line_num+1}: {current_transaction[:80]}...")
                
                current_transaction = ""
                in_transaction = False
                
            elif in_transaction:
                # Continuation line
                if len(line) > 3 and not any(skip in line.lower() for skip in ['page', 'total', 'balance']):
                    current_transaction += " " + line
            if self.debug:

                        print(f"   ? Line {line_num+1}: Appending description - {line[:50]}...")
        
        # Add the last transaction
        if current_transaction and in_transaction:
            combined_lines.append((current_line_num, current_transaction))
            if self.debug:
                print(f"   ?? Final combined transaction {current_line_num+1}: {current_transaction[:80]}...")
        
        return combined_lines
    
    def _parse_real_6month_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a line with Kotak 6-month consolidated format: 
        Date | Narration | Chq/Ref | Withdrawal(Dr)/Deposit(Cr) | Balance(Cr)"""
        try:
            # Find date (DD-MM-YYYY)
            date_match = re.search(r'(\d{2}-\d{2}-\d{4})', line)
            if not date_match:
                return None
            date_str = date_match.group(1)

            # --- Amounts with Dr/Cr ---
            # Handles: "25.00(Dr)", "25(Dr)", "25.00 (Dr)"
            amt_pattern = r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)\s*\((Dr|Cr)\)'
            amt_matches = list(re.finditer(amt_pattern, line, re.IGNORECASE))

            amounts = []
            positions = []
            indicators = []

            for m in amt_matches:
                try:
                    val = float(m.group(1).replace(",", ""))
                    amounts.append(val)
                    positions.append(m.start())
                    indicators.append(m.group(2).capitalize())
                except:
                    continue

            # --- Fallback if regex didn’t catch both amounts ---
            if len(amounts) < 2:
                all_amounts = re.finditer(r'(\d{1,3}(?:,\d{3})*(?:\.\d{1,2})?)', line)
                for m in all_amounts:
                    val = float(m.group(1).replace(",", ""))
                    if val not in amounts:
                        amounts.append(val)
                        positions.append(m.start())
                        # Guess Dr/Cr based on context
                        ctx = line[max(0, m.start()-15):m.end()+15].lower()
                        if "dr" in ctx and "cr" not in ctx:
                            indicators.append("Dr")
                        elif "cr" in ctx and "dr" not in ctx:
                            indicators.append("Cr")
                        else:
                            indicators.append("Cr")  # assume balance by default

            if len(amounts) < 2:
                return None

            # Sort by position in line
            sorted_data = sorted(zip(positions, amounts, indicators), key=lambda x: x[0])
            sorted_amounts = [a for _, a, _ in sorted_data]
            sorted_ind = [ind for _, _, ind in sorted_data]

            txn_amt = sorted_amounts[0]
            bal_amt = sorted_amounts[-1]

            # Apply debit/credit sign
            if sorted_ind[0] == "Dr":
                txn_amt = -abs(txn_amt)
            else:
                txn_amt = abs(txn_amt)

            # --- Description ---
            date_end = line.find(date_str) + len(date_str)
            first_amt_pos = min(positions)
            description = line[date_end:first_amt_pos].strip()
            description = re.sub(r'\s+', ' ', description)

            # Parse date
            try:
                date_obj = datetime.strptime(date_str, "%d-%m-%Y")
                date = date_obj.strftime("%Y-%m-%d")
            except:
                date = date_str

            return {
                "date": date,
                "description": description,
                "amount": txn_amt,
                "balance": bal_amt,
                "ref": "",
                "category": self._categorize_transaction(description, txn_amt),
                "parser_confidence": "high"
            }

        except Exception as e:
            if self.debug:
                print(f"⚠️ Error parsing real 6-month line: {e}")
            return None

    
    def _parse_separate_fields_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a line with separate withdrawal/deposit fields (like 6-month Kotak)."""
        try:
            # Pattern: Date + Description + Withdrawal + Deposit + Balance
            # Example: "01-01-2025 NACH - MUT - DR - INDIAN CLEARING CORP- NACHDR01012506133000 1,200.00 64,545.98"
            
            # Find date (DD-MM-YYYY format)
            date_match = re.search(r'(\d{2}-\d{2}-\d{4})', line)
            if not date_match:
                return None
            
            date_str = date_match.group(1)
            

            # Find amounts
            amount_patterns = [
                r'(\d[\d,]*\.\d{2})',
            ]

            
            amounts = []
            amount_positions = []
            amount_strings = []
            
            for pattern in amount_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    try:
                        full_match = match.group(0)
                        

                        # Skip UPI references
                        if len(full_match.replace(',', '').replace('.', '')) > 8:
                            continue
                        
                        amount_strings.append(full_match)
                        amount_val = float(full_match.replace(',', ''))
                        amounts.append(amount_val)
                        amount_positions.append(match.start())
                    except ValueError:
                        continue
            
            if len(amounts) < 2:
                return None
            

            # Sort amounts by position
            sorted_data = sorted(zip(amount_positions, amounts, amount_strings))
            sorted_amounts = [amount for _, amount, _ in sorted_data]
            sorted_strings = [string for _, _, string in sorted_data]
            
            balance = sorted_amounts[-1]  # Last amount is always balance
            

            # Determine transaction amount
            description = line[date_match.end():min(amount_positions)].strip()
            description_lower = description.lower()
            
            if 'withdrawal' in description_lower or 'dr' in description_lower:
                    transaction_amount = -abs(sorted_amounts[0])  # First amount is withdrawal
            elif 'deposit' in description_lower or 'cr' in description_lower:
                    transaction_amount = abs(sorted_amounts[0])  # First amount is deposit
            else:

                # Assume first amount is transaction
                    if abs(sorted_amounts[0]) < abs(balance):
                        transaction_amount = sorted_amounts[0]
                    else:
                        transaction_amount = sorted_amounts[1]
            
            # Clean description
            description = re.sub(r'\s+', ' ', description).strip()
            
            # Parse date
            try:
                date_obj = datetime.strptime(date_str, '%d-%m-%Y')
                date = date_obj.strftime('%Y-%m-%d')
            except ValueError:
                date = date_str
            
            if self.debug:
                print(f"?? Parsed: Date='{date}', Amount='{transaction_amount}', Balance='{balance}'")
                print(f"?? Description: '{description}'")
            
            return {
                'date': date,
                'description': description,
                'amount': transaction_amount,
                'balance': balance,
                'ref': '',
                'category': self._categorize_transaction(description, transaction_amount),
                'parser_confidence': 'high'
            }
            
        except Exception as e:
            if self.debug:
                print(f"? Error parsing separate fields line: {e}")
            return None

    def _parse_single_month_line(self, line: str) -> Optional[Dict[str, Any]]:
        """Parse a line with single month format (Date + Details + Credit/Debit + Balance)."""
        try:
            # Find date
            date_match = re.search(r'(\d{1,2}\s+[A-Za-z]+\s*,\s*\d{4})', line)
            if not date_match:
                return None
            
            date_str = date_match.group(1)

            # Find credit/debit amounts
            credit_debit_patterns = [
                r'[+−-]\s*(\d[\d,]*\.\d{2})',   # signed amount with 2 decimals
                r'[+−-]\s*(\d[\d,]*\.\d{1})',   # signed amount with 1 decimal
                r'[+−-]\s*(\d[\d,]*)',          # signed whole number
            ]

            
            credit_debit_amounts = []
            credit_debit_strings = []
            
            for pattern in credit_debit_patterns:
                matches = re.finditer(pattern, line)
                for match in matches:
                    try:
                        full_match = match.group(0)
                        amount_val = float(match.group(1).replace(',', ''))
                        
                        # Skip UPI references and phone numbers - IMPROVED LOGIC
                        amount_digits = match.group(1).replace(',', '').replace('.', '')
                        if len(amount_digits) > 10:  # More than 10 digits is suspicious
                            continue
                        if re.match(r'^\d{10}$', amount_digits):  # Exactly 10 digits = phone number
                            continue
                        
                        credit_debit_strings.append(full_match)
                        credit_debit_amounts.append(amount_val)
                    except ValueError:
                        continue

            if not credit_debit_amounts:
                return None

            # Extract description
            date_end = line.find(date_str) + len(date_str)
            first_amount_pos = min([line.find(s) for s in credit_debit_strings if line.find(s) != -1])
            
            description = line[date_end:first_amount_pos].strip()
            description = re.sub(r'\s+', ' ', description).strip()
            description = re.sub(r'^[-\s]+', '', description)
            description = re.sub(r'[-\s]+$', '', description)
            
            # PHANTOM ENTRY DETECTION: More balanced validation
            has_date = bool(date_match)
            has_amount = bool(credit_debit_amounts)
            has_description = bool(description.strip() and len(description.strip()) > 2)  # At least 3 chars
            has_credit_debit = bool(any('+' in s or '-' in s for s in credit_debit_strings))
            
            # Additional check: Description should not be just generic words
            if has_description:
                generic_words = ['other', 'misc', 'miscellaneous', 'various', 'general', 'unknown', 'na', 'n/a']
                if description.lower().strip() in generic_words:
                    has_description = False
                    if self.debug:
                        print(f"   🚫 Phantom entry detected: Generic description '{description}' in line: {line[:80]}...")
            
            # Accept if we have the essential elements: date + amount + credit/debit
            if not (has_date and has_amount and has_credit_debit):
                missing_elements = []
                if not has_date: missing_elements.append("date")
                if not has_amount: missing_elements.append("amount")
                if not has_credit_debit: missing_elements.append("credit/debit indicator")
                
                if self.debug:
                    print(f"   🚫 Phantom entry detected: Missing essential elements ({', '.join(missing_elements)}) in line: {line[:80]}...")
                return None
            
            # Warning for very short descriptions but don't reject
            if not has_description:
                if self.debug:
                    print(f"   ⚠️  Warning: Very short description '{description}' in line: {line[:80]}...")
            
            # Determine transaction amount
            amount_str = credit_debit_strings[0]
            if amount_str.startswith('+'):
                transaction_amount = credit_debit_amounts[0]
            else:
                transaction_amount = -credit_debit_amounts[0]

            # IMPROVED BALANCE EXTRACTION - More intelligent balance detection
            balance = None
            
            # Strategy 1: Look for balance after the transaction amount
            transaction_end_pos = line.find(amount_str) + len(amount_str)
            remaining_line = line[transaction_end_pos:]
            
            # Find all amounts in the remaining line (after transaction amount)
            balance_patterns = [
                r'(\d[\d,]*\.\d{2})',   # with 2 decimals
                r'(\d[\d,]*\.\d{1})',   # with 1 decimal
                r'(\d[\d,]*)',          # whole number
            ]

            
            for pattern in balance_patterns:
                matches = re.finditer(pattern, remaining_line)
                for match in matches:
                    try:
                        amount_val = float(match.group(1).replace(',', ''))
                        
                        # Skip if this is the same as transaction amount
                        if abs(amount_val - abs(transaction_amount)) < 0.01:
                            continue
                        
                        # Skip unreasonably large amounts (more than 1 crore)
                        if amount_val > 10000000:
                            continue
                        
                        # Skip if amount is too close to transaction amount (likely same value)
                        if abs(amount_val - abs(transaction_amount)) < 50:
                            continue
                        
                        # This looks like a valid balance
                        balance = amount_val
                        if self.debug:
                            print(f"   🔍 Found balance after transaction: {balance}")
                        break
                    except ValueError:
                        continue
                if balance is not None:
                    break
            
            # Strategy 2: If no balance found after transaction, look for the last reasonable amount in the line
            if balance is None:
                all_amounts = []
                all_positions = []
                
                for pattern in balance_patterns:
                    matches = re.finditer(pattern, line)
                    for match in matches:
                        try:
                            amount_val = float(match.group(1).replace(',', ''))
                            
                            # Skip if this is the same as transaction amount
                            if abs(amount_val - abs(transaction_amount)) < 0.01:
                                continue
                            
                            # Skip unreasonably large amounts (more than 1 crore)
                            if amount_val > 10000000:
                                continue
                            
                            # Skip if amount is too close to transaction amount
                            if abs(amount_val - abs(transaction_amount)) < 50:
                                continue
                            
                            all_amounts.append(amount_val)
                            all_positions.append(match.start())
                        except ValueError:
                            continue
                
                # If we found multiple amounts, take the last one as balance
                if all_amounts:
                    # Sort by position in line and take the last one
                    sorted_amounts = sorted(zip(all_amounts, all_positions), key=lambda x: x[1])
                    last_amount = sorted_amounts[-1][0]
                    balance = last_amount
                    if self.debug:
                        print(f"   🔍 Using last reasonable amount as balance: {balance}")
            
            # Strategy 3: Use enhanced balance detector as final fallback
            if balance is None:
                balance = self.balance_detector.extract_balance_improved(line, transaction_amount)
                if balance is not None and self.debug:
                    print(f"   🔍 Enhanced detector found balance: {balance}")
            
            # Final validation: ensure balance is reasonable
            if balance is not None:
                # Skip if balance is same as transaction amount
                if abs(balance - abs(transaction_amount)) < 0.01:
                    balance = None
                # Skip if balance is unreasonably large (more than 1 crore)
                elif balance > 10000000:
                    balance = None
                    if self.debug:
                        print(f"   ⚠️  Balance {balance} too large, setting to None")
            


                

                
            #     if balance is None:
            #         all_amounts = []
            #         all_positions = []
            #         for pattern in balance_patterns:
            #             matches = re.finditer(pattern, line)
            #             for match in matches:
            #                 try:
            #                     amount_val = float(match.group(1).replace(',', ''))
            #                     amount_digits = match.group(1).replace(',', '').replace('.', '')
            #                     if abs(amount_val - abs(transaction_amount)) < 0.01:
            #                         continue
            #                     if amount_val < 1000 or amount_val > 10000000:
            #                         continue
            #                     if len(amount_digits) > 8:
            #                         continue
            #                     if abs(amount_val - abs(transaction_amount)) < 100:
            #                         continue
            #                     all_amounts.append(amount_val)
            #                     all_positions.append(match.start())
            #                 except ValueError:
            #                     continue
                    
                    # if all_amounts:
                    #     sorted_amounts = sorted(zip(all_amounts, all_positions), key=lambda x: x[1])
                    #     last_amount = sorted_amounts[-1][0]
                    #     if abs(last_amount - abs(transaction_amount)) > 500:
                    #         balance = last_amount
                    #         if self.debug:
                    #             print(f"   🔍 Using last reasonable amount as balance: {balance}")
                    #     else:
                    #         if self.debug:
                    #             print(f"   ⚠️  Last amount {last_amount} too close to transaction {transaction_amount}, skipping")
            
            # Handle special cases
            description_lower = description.lower()
            if 'opening balance' in description_lower:
                balance = abs(transaction_amount)
                if self.debug:
                    print(f"   🔄 Opening balance: using transaction amount as balance: {balance}")
            
            if 'opening balance' not in description_lower:
                skip_patterns = [
                    'date', 'transaction details', 'cheque/reference#', 'debit', 'credit', 'balance',
                    'narration', 'withdrawal', 'deposit', 'chq/ref', 'total debited', 'total credited', 
                    'total transactions', 'summary', 'closing balance', 'statement period', 'page', 
                    'generated on', 'overdraft drawing power', 'others', 'sweep td broken'
                ]
                if any(pattern in description_lower for pattern in skip_patterns):
                    if self.debug:
                        print(f"   🚫 Skipping summary row: '{description}'")
                    return None
                
                # PHANTOM ENTRY DETECTION: Only reject lines that are clearly non-transactional
                # Keep all valid transaction descriptions including "Other", "Misc", etc.
                pass
            
            try:
                clean_date_str = re.sub(r'\s*,\s*', ', ', date_str)
                date_obj = datetime.strptime(clean_date_str, '%d %b, %Y')
                date = date_obj.strftime('%Y-%m-%d')
            except ValueError:
                date = date_str
            
            if self.debug:
                print(f"✅ Parsed: Date='{date}', Amount='{amount_str}' -> {transaction_amount}, Balance='{balance}'")
                print(f"✅ Description: '{description}'")
            
            return {
                'date': date,
                'description': description,
                'amount': transaction_amount,
                'balance': balance,
                'ref': '',
                'category': self._categorize_transaction(description, transaction_amount),
                'parser_confidence': 'high'
            }
            
        except Exception as e:
            if self.debug:
                print(f"❌ Error parsing single month line: {e}")
            return None

    

    def _table_parser(self, raw_text: str) -> List[Dict[str, Any]]:
        """Parse traditional table format statements."""
        if self.debug:

            print("   ?? Using table parser...")
        
        # Table parser for traditional formats
        # This handles structured table data
        return self._single_month_parser(raw_text)
    
    def _generic_parser(self, raw_text: str) -> List[Dict[str, Any]]:
        """Generic parser as fallback - tries all available parsers."""
        if self.debug:
            print("   ?? Using generic parser as fallback...")
        
        # Try all available parsers and combine results
        all_transactions = []
        
        # Try single month parser
        transactions = self._single_month_parser(raw_text)
        if transactions:
            all_transactions.extend(transactions)
            if self.debug:
                print(f"   ?? Single month parser found {len(transactions)} transactions")
        
        # Try real 6-month parser
        transactions = self._real_6month_parser(raw_text)
        if transactions:
            all_transactions.extend(transactions)
            if self.debug:
                print(f"   ?? Real 6-month parser found {len(transactions)} transactions")
        
        # Try separate fields parser
        transactions = self._separate_fields_parser(raw_text)
        if transactions:
            all_transactions.extend(transactions)
            if self.debug:
                print(f"   ?? Separate fields parser found {len(transactions)} transactions")
        
        # Try consolidated parser
        transactions = self._consolidated_parser(raw_text)
        if transactions:
            all_transactions.extend(transactions)
            if self.debug:
                print(f"   ?? Consolidated parser found {len(transactions)} transactions")
        
        # Try detailed parser
        transactions = self._detailed_parser(raw_text)
        if transactions:
            all_transactions.extend(transactions)
            if self.debug:
                print(f"   ?? Detailed parser found {len(transactions)} transactions")
        
        # Remove duplicates and return
        unique_transactions = self._remove_duplicates(all_transactions)
        if self.debug:
            print(f"   ?? Total unique transactions: {len(unique_transactions)}")
        
        return unique_transactions
    
    def _remove_duplicates(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Remove duplicate transactions based on date, description, and amount."""
        seen = set()
        unique_transactions = []
        
        for transaction in transactions:
            # Create a unique key
            key = (
                transaction.get('date', ''),
                transaction.get('description', '')[:50],  # First 50 chars of description
                transaction.get('amount', 0)
            )
            
            if key not in seen:
                seen.add(key)
                unique_transactions.append(transaction)
        
        return unique_transactions
    

    def _calculate_confidence(self, transactions: List[Dict[str, Any]], layout_info: Dict) -> float:
        """Calculate confidence score for the parsing results."""
        if not transactions:
            return 0.0
        
        # Base confidence on layout detection
        base_confidence = min(layout_info.get('score', 0) / 3.0, 1.0)
        
        # Boost confidence if we found many transactions
        transaction_boost = min(len(transactions) / 50.0, 0.3)  # Max 0.3 boost for 50+ transactions
        
        # Boost confidence if transactions have good data quality
        quality_boost = 0.0
        valid_transactions = 0
        
        for transaction in transactions:
            if (transaction.get('date') and 
                transaction.get('description') and 
                transaction.get('amount') is not None and
                transaction.get('balance') is not None):
                valid_transactions += 1
        
        if transactions:
            quality_boost = (valid_transactions / len(transactions)) * 0.2  # Max 0.2 boost for quality
        
        total_confidence = base_confidence + transaction_boost + quality_boost
        return min(total_confidence, 1.0)
    
    def _enhance_transactions(self, transactions: List[Dict[str, Any]], bank_type: str) -> List[Dict[str, Any]]:
        """Enhance transactions with additional metadata."""
        for transaction in transactions:
            transaction['bank_type'] = bank_type.upper()
            transaction['extraction_method'] = 'layout_aware_parser'
            
            if 'ref' not in transaction:
                transaction['ref'] = ''
            if 'category' not in transaction:
                transaction['category'] = 'Other'
        
        # Sort by date
        transactions.sort(key=lambda x: x.get('date', ''))
        return transactions
    
    def _categorize_transaction(self, description: str, amount: float) -> str:
        """Categorize transaction based on description and amount."""
        desc_lower = description.lower()
        
        if 'nach' in desc_lower:
            return 'NACH'
        if any(k in desc_lower for k in ['upi', 'gpay', 'google pay', 'phonepe', 'paytm']):
            return 'UPI'
        if 'neft' in desc_lower:
            return 'NEFT'
        if any(k in desc_lower for k in ['atm', 'cash withdrawal']):
            return 'ATM'
        if any(k in desc_lower for k in ['cheque', 'chq', 'check']):
            return 'Cheque'
        
        return 'Other'


    def _validate_transaction(self, transaction: Dict[str, Any]) -> bool:
        """Validate that a transaction has all required fields."""
        required_fields = ['date', 'description', 'amount']
        return all(field in transaction and transaction[field] is not None for field in required_fields)
    
    def _clean_transaction_data(self, transaction: Dict[str, Any]) -> Dict[str, Any]:
        """Clean and normalize transaction data."""
        if 'description' in transaction:
            transaction['description'] = re.sub(r'\s+', ' ', transaction['description']).strip()
        
        if 'date' in transaction and isinstance(transaction['date'], str):
            # Ensure date is in YYYY-MM-DD format
            try:
                if re.match(r'\d{4}-\d{2}-\d{2}', transaction['date']):
                    pass  # Already in correct format
                elif re.match(r'\d{1,2}\s+[A-Za-z]+\s*,\s*\d{4}', transaction['date']):
                    # Convert "01 May, 2025" to "2025-05-01"
                    clean_date_str = re.sub(r'\s*,\s*', ', ', transaction['date'])
                    date_obj = datetime.strptime(clean_date_str, '%d %b, %Y')
                    transaction['date'] = date_obj.strftime('%Y-%m-%d')
                elif re.match(r'\d{2}-\d{2}-\d{4}', transaction['date']):
                    # Convert "01-05-2025" to "2025-05-01"
                    date_obj = datetime.strptime(transaction['date'], '%d-%m-%Y')
                    transaction['date'] = date_obj.strftime('%Y-%m-%d')
            except ValueError:
                pass  # Keep original if parsing fails
        
        return transaction
    
    def _merge_similar_transactions(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Merge transactions that appear to be the same but were split."""
        if len(transactions) < 2:
            return transactions
        
        merged = []
        i = 0
        
        while i < len(transactions):
            current = transactions[i]
            merged_transaction = current.copy()
            
            # Look ahead for similar transactions
            j = i + 1
            while j < len(transactions):
                next_transaction = transactions[j]
                
                # Check if transactions are similar (same date, similar description, same amount)
                if (current['date'] == next_transaction['date'] and
                    current['amount'] == next_transaction['amount'] and
                    self._are_descriptions_similar(current['description'], next_transaction['description'])):
                    
                    # Merge descriptions
                    merged_transaction['description'] = f"{current['description']} | {next_transaction['description']}"
                    merged_transaction['merged_count'] = merged_transaction.get('merged_count', 1) + 1
                    
                    # Skip the merged transaction
                    j += 1
                else:
                    break
            
            merged.append(merged_transaction)
            i = j
        
        return merged
    
    def _are_descriptions_similar(self, desc1: str, desc2: str) -> bool:
        """Check if two transaction descriptions are similar."""
        if not desc1 or not desc2:
            return False
        
        # Simple similarity check - if they share common words
        words1 = set(desc1.lower().split())
        words2 = set(desc2.lower().split())
        
        common_words = words1.intersection(words2)
        total_words = words1.union(words2)
        
        if not total_words:
            return False
        
        similarity = len(common_words) / len(total_words)
        return similarity > 0.3  # 30% similarity threshold
    
    def _extract_additional_info(self, raw_text: str) -> Dict[str, Any]:
        """Extract additional information from the statement."""
        info = {
            'account_number': None,
            'account_holder': None,
            'statement_period': None,
            'opening_balance': None,
            'closing_balance': None,
            'total_credits': 0,
            'total_debits': 0
        }
        
        # Extract account number
        account_match = re.search(r'account\s*[#:]?\s*(\d{9,16})', raw_text, re.IGNORECASE)
        if account_match:
            info['account_number'] = account_match.group(1)
        
        # Extract account holder name
        name_match = re.search(r'account\s*holder\s*[#:]?\s*([A-Za-z\s]+)', raw_text, re.IGNORECASE)
        if name_match:
            info['account_holder'] = name_match.group(1).strip()
        
        # Extract statement period
        period_match = re.search(r'statement\s*period\s*[#:]?\s*([A-Za-z\s\d,]+)', raw_text, re.IGNORECASE)
        if period_match:
            info['statement_period'] = period_match.group(1).strip()
        
        # Extract opening balance
        opening_match = re.search(r'opening\s*balance\s*[#:]?\s*([+-]?\d{1,3}(?:,\d{3})*\.?\d*)', raw_text, re.IGNORECASE)
        if opening_match:
            try:
                info['opening_balance'] = float(opening_match.group(1).replace(',', ''))
            except ValueError:
                pass
        
        # Extract closing balance
        closing_match = re.search(r'closing\s*balance\s*[#:]?\s*([+-]?\d{1,3}(?:,\d{3})*\.?\d*)', raw_text, re.IGNORECASE)
        if closing_match:
            try:
                info['closing_balance'] = float(closing_match.group(1).replace(',', ''))
            except ValueError:
                pass
        
        return info
    
    def _calculate_totals(self, transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate totals from transactions."""
        totals = {
            'total_credits': 0.0,
            'total_debits': 0.0,
            'net_amount': 0.0,
            'transaction_count': len(transactions)
        }
        
        for transaction in transactions:
            amount = transaction.get('amount', 0)
            if amount > 0:
                totals['total_credits'] += amount
            else:
                totals['total_debits'] += abs(amount)
        
        totals['net_amount'] = totals['total_credits'] - totals['total_debits']
        return totals
    
    def _generate_summary_report(self, transactions: List[Dict[str, Any]], parsing_info: Dict, additional_info: Dict) -> Dict[str, Any]:
        """Generate a comprehensive summary report."""
        totals = self._calculate_totals(transactions)
        
        report = {
            'parsing_summary': parsing_info,
            'transaction_summary': totals,
            'additional_info': additional_info,
            'quality_metrics': {
                'completeness': self._calculate_completeness(transactions),
                'accuracy': self._calculate_accuracy(transactions),
                'consistency': self._calculate_consistency(transactions)
            },
            'recommendations': self._generate_recommendations(transactions, parsing_info)
        }
        
        return report
    
    def _calculate_completeness(self, transactions: List[Dict[str, Any]]) -> float:
        """Calculate completeness score of extracted data."""
        if not transactions:
            return 0.0
        
        required_fields = ['date', 'description', 'amount', 'balance']
        complete_transactions = 0
        
        for transaction in transactions:
            if all(field in transaction and transaction[field] is not None for field in required_fields):
                complete_transactions += 1
        
        return complete_transactions / len(transactions)
    
    def _calculate_accuracy(self, transactions: List[Dict[str, Any]]) -> float:
        """Calculate accuracy score of extracted data."""
        if not transactions:
            return 0.0
        
        # Simple accuracy check - transactions with reasonable amounts
        reasonable_transactions = 0
        
        for transaction in transactions:
            amount = transaction.get('amount', 0)
            balance = transaction.get('balance', 0)
            
            # Check if amounts are reasonable (not too large or negative when they shouldn't be)
            if (isinstance(amount, (int, float)) and 
                isinstance(balance, (int, float)) and
                abs(amount) < 10000000 and  # Less than 10 crore
                balance >= 0):  # Balance should be positive
                reasonable_transactions += 1
        
        return reasonable_transactions / len(transactions)
    
    def _calculate_consistency(self, transactions: List[Dict[str, Any]]) -> float:
        """Calculate consistency score of extracted data."""
        if len(transactions) < 2:
            return 1.0
        
        # Check date consistency
        dates = [t.get('date') for t in transactions if t.get('date')]
        if not dates:
            return 0.0
        
        # Check if dates are in chronological order
        try:
            sorted_dates = sorted(dates)
            if dates == sorted_dates:
                return 1.0
            else:
                return 0.5  # Partial consistency
        except:
            return 0.0
    
    def _generate_recommendations(self, transactions: List[Dict[str, Any]], parsing_info: Dict) -> List[str]:
        """Generate recommendations based on parsing results."""
        recommendations = []
        
        if parsing_info.get('confidence_score', 0) < 0.7:
            recommendations.append("Low confidence in parsing results. Consider manual review.")
        
        if len(transactions) < 10:
            recommendations.append("Few transactions extracted. Check if statement format is supported.")
        
        # Check for potential issues
        missing_balances = sum(1 for t in transactions if t.get('balance') is None)
        if missing_balances > len(transactions) * 0.1:  # More than 10% missing
            recommendations.append("Many transactions missing balance information.")
        
        # Check for duplicate transactions
        seen = set()
        duplicates = 0
        for t in transactions:
            key = (t.get('date'), t.get('description')[:30], t.get('amount'))
            if key in seen:
                duplicates += 1
            seen.add(key)
        
        if duplicates > 0:
            recommendations.append(f"Found {duplicates} potential duplicate transactions.")
        
        if not recommendations:
            recommendations.append("Parsing completed successfully with high confidence.")
        
        return recommendations
    
    def _handle_special_cases(self, raw_text: str) -> List[Dict[str, Any]]:
        """Handle special cases and edge cases in parsing."""
        special_cases = []
        
        # Handle empty or very short text
        if not raw_text or len(raw_text.strip()) < 50:
            if self.debug:
                print("   ?? Text too short, attempting minimal parsing...")
            return []
        
        # Handle encrypted or corrupted text
        if 'encrypted' in raw_text.lower() or 'password' in raw_text.lower():
            if self.debug:
                print("   ?? Text appears to be encrypted or password protected...")
            return []
        
        # Handle text with only numbers or special characters
        if re.match(r'^[\d\s\-\+\.\,]+$', raw_text.strip()):
            if self.debug:
                print("   ?? Text contains only numbers and symbols...")
            return []
        
        return special_cases
    
    def _optimize_parsing_performance(self, raw_text: str) -> str:
        """Optimize text for better parsing performance."""
        # Remove excessive whitespace
        optimized_text = re.sub(r'\s+', ' ', raw_text)
        
        # Remove common noise patterns
        noise_patterns = [
            r'page\s+\d+\s+of\s+\d+',  # Page numbers
            r'statement\s+generated\s+on\s+.*',  # Generation timestamps
            r'this\s+is\s+a\s+computer\s+generated\s+statement',  # Computer generated notices
            r'please\s+retain\s+this\s+statement',  # Retention notices
        ]
        
        for pattern in noise_patterns:
            optimized_text = re.sub(pattern, '', optimized_text, flags=re.IGNORECASE)
        
        return optimized_text.strip()
    
    def _validate_parsing_results(self, transactions: List[Dict[str, Any]]) -> Tuple[bool, List[str]]:
        """Validate parsing results and return issues found."""
        issues = []
        
        if not transactions:
            issues.append("No transactions extracted")
            return False, issues
        
        # Check for basic data quality
        for i, transaction in enumerate(transactions):
            if not transaction.get('date'):
                issues.append(f"Transaction {i+1}: Missing date")
            
            if not transaction.get('description'):
                issues.append(f"Transaction {i+1}: Missing description")
            
            if transaction.get('amount') is None:
                issues.append(f"Transaction {i+1}: Missing amount")
            
            if transaction.get('balance') is None:
                issues.append(f"Transaction {i+1}: Missing balance")
        
        # Check for data consistency
        if len(issues) == 0:
            return True, ["All transactions validated successfully"]
        
        return len(issues) < len(transactions), issues
    
    def _generate_debug_info(self, raw_text: str, transactions: List[Dict[str, Any]], parsing_info: Dict) -> Dict[str, Any]:
        """Generate comprehensive debug information."""
        debug_info = {
            'text_analysis': {
                'total_length': len(raw_text),
                'line_count': len(raw_text.split('\n')),
                'word_count': len(raw_text.split()),
                'contains_dates': bool(re.search(r'\d{1,2}[-/\s][A-Za-z]{3,4}[-/\s]\d{4}', raw_text)),
                'contains_amounts': bool(re.search(r'[+-]?\d{1,3}(?:,\d{3})*\.?\d*', raw_text)),
                'contains_keywords': {
                    'nach': 'nach' in raw_text.lower(),
                    'neft': 'neft' in raw_text.lower(),
                    'upi': 'upi' in raw_text.lower(),
                    'atm': 'atm' in raw_text.lower()
                }
            },
            'parsing_details': parsing_info,
            'transaction_analysis': {
                'total_transactions': len(transactions),
                'date_range': self._get_date_range(transactions),
                'amount_range': self._get_amount_range(transactions),
                'categories': self._get_category_distribution(transactions)
            },
            'performance_metrics': {
                'parsing_time': getattr(self, '_parsing_time', 0),
                'memory_usage': getattr(self, '_memory_usage', 0)
            }
        }
        
        return debug_info
    
    def _get_date_range(self, transactions: List[Dict[str, Any]]) -> Dict[str, str]:
        """Get the date range of transactions."""
        if not transactions:
            return {'start': None, 'end': None}
        
        dates = [t.get('date') for t in transactions if t.get('date')]
        if not dates:
            return {'start': None, 'end': None}
        
        try:
            sorted_dates = sorted(dates)
            return {'start': sorted_dates[0], 'end': sorted_dates[-1]}
        except:
            return {'start': None, 'end': None}
    
    def _get_amount_range(self, transactions: List[Dict[str, Any]]) -> Dict[str, float]:
        """Get the amount range of transactions."""
        if not transactions:
            return {'min': 0, 'max': 0, 'avg': 0}
        
        amounts = [abs(t.get('amount', 0)) for t in transactions if t.get('amount') is not None]
        if not amounts:
            return {'min': 0, 'max': 0, 'avg': 0}
        
        return {
            'min': min(amounts),
            'max': max(amounts),
            'avg': sum(amounts) / len(amounts)
        }
    
    def _get_category_distribution(self, transactions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Get the distribution of transaction categories."""
        categories = {}
        for transaction in transactions:
            category = transaction.get('category', 'Other')
            categories[category] = categories.get(category, 0) + 1
        return categories
    
    def _enhance_transaction_metadata(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Enhance transactions with additional metadata and insights."""
        for transaction in transactions:
            # Add transaction type classification
            amount = transaction.get('amount', 0)
            if amount > 0:
                transaction['transaction_type'] = 'Credit'
                transaction['is_credit'] = True
                transaction['is_debit'] = False
            else:
                transaction['transaction_type'] = 'Debit'
                transaction['is_credit'] = False
                transaction['is_debit'] = True
            
            # Add amount magnitude classification
            abs_amount = abs(amount)
            if abs_amount < 1000:
                transaction['amount_category'] = 'Small (<1K)'
            elif abs_amount < 10000:
                transaction['amount_category'] = 'Medium (1K-10K)'
            elif abs_amount < 100000:
                transaction['amount_category'] = 'Large (10K-1L)'
            else:
                transaction['amount_category'] = 'Very Large (>1L)'
            
            # Add time-based metadata
            if transaction.get('date'):
                try:
                    date_obj = datetime.strptime(transaction['date'], '%Y-%m-%d')
                    transaction['day_of_week'] = date_obj.strftime('%A')
                    transaction['month'] = date_obj.strftime('%B')
                    transaction['quarter'] = f"Q{(date_obj.month - 1) // 3 + 1}"
                except:
                    pass
            
            # Add fraud detection flags
            transaction['fraud_indicators'] = self._detect_fraud_indicators(transaction)
        
        return transactions
    
    def _detect_fraud_indicators(self, transaction: Dict[str, Any]) -> List[str]:
        """Detect potential fraud indicators in a transaction."""
        indicators = []
        
        # Check for unusual amounts
        amount = abs(transaction.get('amount', 0))
        if amount > 1000000:  # More than 10 lakhs
            indicators.append('High Value Transaction')
        
        # Check for unusual timing
        if transaction.get('date'):
            try:
                date_obj = datetime.strptime(transaction['date'], '%Y-%m-%d')
                if date_obj.weekday() >= 5:  # Weekend
                    indicators.append('Weekend Transaction')
                if date_obj.hour < 6 or date_obj.hour > 22:  # Late night
                    indicators.append('Late Night Transaction')
            except:
                pass
        
        # Check for suspicious descriptions
        description = transaction.get('description', '').lower()
        suspicious_keywords = ['urgent', 'emergency', 'quick', 'fast', 'immediate', 'urgently']
        if any(keyword in description for keyword in suspicious_keywords):
            indicators.append('Suspicious Description')
        
        # Check for round amounts (potential fraud indicator)
        if amount > 0 and amount % 1000 == 0:
            indicators.append('Round Amount')
        
        return indicators
    
    def _generate_transaction_summary(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive transaction summary with insights."""
        if not transactions:
            return {}
        
        summary = {
            'total_transactions': len(transactions),
            'total_credits': 0,
            'total_debits': 0,
            'net_flow': 0,
            'average_transaction': 0,
            'largest_transaction': 0,
            'smallest_transaction': float('inf'),
            'transaction_types': {'credit': 0, 'debit': 0},
            'amount_categories': {},
            'daily_patterns': {},
            'monthly_patterns': {},
            'category_breakdown': {},
            'fraud_risk_score': 0
        }
        
        total_amount = 0
        fraud_indicators_count = 0
        
        for transaction in transactions:
            amount = transaction.get('amount', 0)
            abs_amount = abs(amount)
            
            # Basic totals
            if amount > 0:
                summary['total_credits'] += amount
                summary['transaction_types']['credit'] += 1
            else:
                summary['total_debits'] += abs_amount
                summary['transaction_types']['debit'] += 1
            
            total_amount += abs_amount
            
            # Amount tracking
            summary['largest_transaction'] = max(summary['largest_transaction'], abs_amount)
            summary['smallest_transaction'] = min(summary['smallest_transaction'], abs_amount)
            
            # Category breakdown
            category = transaction.get('category', 'Other')
            summary['category_breakdown'][category] = summary['category_breakdown'].get(category, 0) + 1
            
            # Amount categories
            amount_cat = transaction.get('amount_category', 'Unknown')
            summary['amount_categories'][amount_cat] = summary['amount_categories'].get(amount_cat, 0) + 1
            
            # Time patterns
            if transaction.get('day_of_week'):
                day = transaction['day_of_week']
                summary['daily_patterns'][day] = summary['daily_patterns'].get(day, 0) + 1
            
            if transaction.get('month'):
                month = transaction['month']
                summary['monthly_patterns'][month] = summary['monthly_patterns'].get(month, 0) + 1
            
            # Fraud indicators
            fraud_indicators = transaction.get('fraud_indicators', [])
            fraud_indicators_count += len(fraud_indicators)
        
        # Calculate derived metrics
        summary['net_flow'] = summary['total_credits'] - summary['total_debits']
        summary['average_transaction'] = total_amount / len(transactions) if transactions else 0
        summary['fraud_risk_score'] = min(fraud_indicators_count / len(transactions) * 100, 100)
        
        return summary
    
    def _detect_anomalies(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Detect anomalous transactions that might need attention."""
        anomalies = []
        
        if len(transactions) < 3:
            return anomalies
        
        # Calculate statistical measures
        amounts = [abs(t.get('amount', 0)) for t in transactions]
        mean_amount = sum(amounts) / len(amounts)
        variance = sum((x - mean_amount) ** 2 for x in amounts) / len(amounts)
        std_dev = variance ** 0.5
        
        for transaction in transactions:
            amount = abs(transaction.get('amount', 0))
            anomaly_score = 0
            anomaly_reasons = []
            
            # Statistical anomalies (amounts far from mean)
            if std_dev > 0:
                z_score = abs(amount - mean_amount) / std_dev
                if z_score > 2.5:  # More than 2.5 standard deviations
                    anomaly_score += 30
                    anomaly_reasons.append(f'Statistical outlier (Z-score: {z_score:.2f})')
            
            # Unusual frequency patterns
            date = transaction.get('date')
            if date:
                same_date_transactions = [t for t in transactions if t.get('date') == date]
                if len(same_date_transactions) > 5:  # More than 5 transactions on same day
                    anomaly_score += 20
                    anomaly_reasons.append(f'High frequency day ({len(same_date_transactions)} transactions)')
            
            # Unusual amount patterns
            if amount > mean_amount * 5:  # 5x the average
                anomaly_score += 25
                anomaly_reasons.append('Amount significantly above average')
            
            # Round number anomalies
            if amount > 0 and amount % 10000 == 0:  # Round to 10K
                anomaly_score += 15
                anomaly_reasons.append('Suspicious round amount')
            
            # Add to anomalies if score is high enough
            if anomaly_score >= 30:
                anomalies.append({
                    'transaction': transaction,
                    'anomaly_score': anomaly_score,
                    'reasons': anomaly_reasons,
                    'severity': 'High' if anomaly_score >= 50 else 'Medium' if anomaly_score >= 30 else 'Low'
                })
        
        # Sort by anomaly score
        anomalies.sort(key=lambda x: x['anomaly_score'], reverse=True)
        return anomalies
    
    def _generate_risk_assessment(self, transactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate a comprehensive risk assessment for the account."""
        risk_assessment = {
            'overall_risk_score': 0,
            'risk_factors': [],
            'risk_level': 'Low',
            'recommendations': [],
            'monitoring_required': False
        }
        
        if not transactions:
            return risk_assessment
        
        risk_score = 0
        risk_factors = []
        
        # High value transaction risk
        high_value_count = sum(1 for t in transactions if abs(t.get('amount', 0)) > 1000000)
        if high_value_count > 0:
            risk_score += high_value_count * 10
            risk_factors.append(f'{high_value_count} high value transactions (>10L)')
        
        # Unusual timing risk
        weekend_transactions = sum(1 for t in transactions if t.get('day_of_week') in ['Saturday', 'Sunday'])
        if weekend_transactions > len(transactions) * 0.3:  # More than 30% on weekends
            risk_score += 20
            risk_factors.append('High weekend transaction volume')
        
        # Category-based risk
        suspicious_categories = ['ATM', 'Cash Withdrawal']
        suspicious_count = sum(1 for t in transactions if t.get('category') in suspicious_categories)
        if suspicious_count > len(transactions) * 0.4:  # More than 40% suspicious categories
            risk_score += 25
            risk_factors.append('High volume of cash transactions')
        
        # Frequency risk
        if len(transactions) > 100:  # Very high transaction volume
            risk_score += 15
            risk_factors.append('Very high transaction volume')
        
        # Amount pattern risk
        amounts = [abs(t.get('amount', 0)) for t in transactions]
        if amounts:
            avg_amount = sum(amounts) / len(amounts)
            if avg_amount > 50000:  # High average transaction amount
                risk_score += 20
                risk_factors.append('High average transaction amount')
        
        # Determine risk level
        if risk_score >= 70:
            risk_level = 'High'
            monitoring_required = True
        elif risk_score >= 40:
            risk_level = 'Medium'
            monitoring_required = True
        else:
            risk_level = 'Low'
            monitoring_required = False
        
        # Generate recommendations
        recommendations = []
        if risk_score >= 70:
            recommendations.append('Immediate account review required')
            recommendations.append('Consider temporary restrictions')
        elif risk_score >= 40:
            recommendations.append('Enhanced monitoring recommended')
            recommendations.append('Review transaction patterns')
        else:
            recommendations.append('Standard monitoring sufficient')
        
        risk_assessment.update({
            'overall_risk_score': risk_score,
            'risk_factors': risk_factors,
            'risk_level': risk_level,
            'recommendations': recommendations,
            'monitoring_required': monitoring_required
        })
        
        return risk_assessment
    
    def _optimize_for_performance(self, raw_text: str) -> str:
        """Advanced text optimization for better parsing performance."""
        # Remove excessive whitespace and normalize
        optimized_text = re.sub(r'\s+', ' ', raw_text)
        
        # Remove common noise patterns that don't affect parsing
        noise_patterns = [
            r'page\s+\d+\s+of\s+\d+',  # Page numbers
            r'statement\s+generated\s+on\s+.*',  # Generation timestamps
            r'this\s+is\s+a\s+computer\s+generated\s+statement',  # Computer generated notices
            r'please\s+retain\s+this\s+statement',  # Retention notices
            r'for\s+queries\s+contact\s+.*',  # Contact information
            r'terms\s+and\s+conditions\s+apply',  # Legal disclaimers
            r'statement\s+period\s+.*',  # Statement period headers
            r'account\s+summary\s+.*',  # Account summary headers
        ]
        
        for pattern in noise_patterns:
            optimized_text = re.sub(pattern, '', optimized_text, flags=re.IGNORECASE)
        
        # Remove excessive punctuation that might interfere with parsing
        optimized_text = re.sub(r'[^\w\s\d\-\.\,\+\/]', ' ', optimized_text)
        
        # Normalize date formats for better recognition
        date_patterns = [
            (r'(\d{1,2})\s*-\s*(\d{1,2})\s*-\s*(\d{4})', r'\1-\2-\3'),  # Normalize DD-MM-YYYY
            (r'(\d{1,2})\s*\/\s*(\d{1,2})\s*\/\s*(\d{4})', r'\1-\2-\3'),  # Convert DD/MM/YYYY to DD-MM-YYYY
        ]
        
        for old_pattern, new_pattern in date_patterns:
            optimized_text = re.sub(old_pattern, new_pattern, optimized_text)
        
        return optimized_text.strip()
    
    def _cache_parsing_results(self, raw_text_hash: str, transactions: List[Dict[str, Any]], parsing_info: Dict) -> None:
        """Cache parsing results for performance optimization."""
        if not hasattr(self, '_parsing_cache'):
            self._parsing_cache = {}
        
        # Simple in-memory cache with hash key
        self._parsing_cache[raw_text_hash] = {
            'transactions': transactions,
            'parsing_info': parsing_info,
            'timestamp': datetime.now(),
            'cache_hits': 0
        }
        
        # Limit cache size to prevent memory issues
        if len(self._parsing_cache) > 100:
            # Remove oldest entries
            oldest_key = min(self._parsing_cache.keys(), 
                           key=lambda k: self._parsing_cache[k]['timestamp'])
            del self._parsing_cache[oldest_key]
    
    def _get_cached_results(self, raw_text_hash: str) -> Optional[Tuple[List[Dict[str, Any]], Dict[str, Any]]]:
        """Retrieve cached parsing results if available."""
        if not hasattr(self, '_parsing_cache'):
            return None
        
        cached = self._parsing_cache.get(raw_text_hash)
        if cached:
            # Update cache hit count
            cached['cache_hits'] += 1
            cached['timestamp'] = datetime.now()
            
            if self.debug:
                print(f"   ?? Using cached results (hit #{cached['cache_hits']})")
            
            return cached['transactions'], cached['parsing_info']
        
        return None
    
    def _generate_hash(self, text: str) -> str:
        """Generate a hash for the input text for caching purposes."""
        import hashlib
        return hashlib.md5(text.encode('utf-8')).hexdigest()
    
    def _monitor_performance(self, start_time: float) -> Dict[str, float]:
        """Monitor parsing performance metrics."""
        import time
        import psutil
        
        end_time = time.time()
        parsing_time = end_time - start_time
        
        # Get memory usage
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        
        performance_metrics = {
            'parsing_time_seconds': parsing_time,
            'memory_usage_mb': memory_mb,
            'transactions_per_second': 0,  # Will be calculated later
            'efficiency_score': 0  # Will be calculated later
        }
        
        # Store for later use
        self._parsing_time = parsing_time
        self._memory_usage = memory_mb
        
        return performance_metrics
    
    def _calculate_efficiency_metrics(self, transactions: List[Dict[str, Any]], performance_metrics: Dict[str, float]) -> Dict[str, float]:
        """Calculate efficiency metrics for the parsing process."""
        if not transactions:
            return performance_metrics
        
        # Calculate transactions per second
        parsing_time = performance_metrics.get('parsing_time_seconds', 0)
        if parsing_time > 0:
            performance_metrics['transactions_per_second'] = len(transactions) / parsing_time
        
        # Calculate efficiency score (higher is better)
        # Factors: speed, memory usage, transaction count
        speed_score = min(performance_metrics.get('transactions_per_second', 0) / 10, 1.0)  # Normalize to 0-1
        memory_score = max(0, 1 - (performance_metrics.get('memory_usage_mb', 0) / 1000))  # Lower memory = higher score
        transaction_score = min(len(transactions) / 100, 1.0)  # More transactions = higher score
        
        efficiency_score = (speed_score * 0.4 + memory_score * 0.3 + transaction_score * 0.3)
        performance_metrics['efficiency_score'] = efficiency_score
        
        return performance_metrics
    
    def _export_advanced_report(self, transactions: List[Dict[str, Any]], parsing_info: Dict, 
                               additional_info: Dict, performance_metrics: Dict[str, float]) -> Dict[str, Any]:
        """Generate an advanced export report with comprehensive analysis."""
        # Generate all analysis components
        transaction_summary = self._generate_transaction_summary(transactions)
        anomalies = self._detect_anomalies(transactions)
        risk_assessment = self._generate_risk_assessment(transactions)
        debug_info = self._generate_debug_info("", transactions, parsing_info)
        
        # Calculate efficiency metrics
        performance_metrics = self._calculate_efficiency_metrics(transactions, performance_metrics)
        
        # Enhanced transactions with metadata
        enhanced_transactions = self._enhance_transaction_metadata(transactions.copy())
        
        # Compile comprehensive report
        advanced_report = {
            'executive_summary': {
                'total_transactions': len(transactions),
                'parsing_confidence': parsing_info.get('confidence_score', 0),
                'overall_risk_level': risk_assessment.get('risk_level', 'Unknown'),
                'parsing_efficiency': performance_metrics.get('efficiency_score', 0)
            },
            'transaction_analysis': transaction_summary,
            'risk_assessment': risk_assessment,
            'anomaly_detection': {
                'total_anomalies': len(anomalies),
                'high_risk_anomalies': len([a for a in anomalies if a.get('severity') == 'High']),
                'anomalies': anomalies
            },
            'performance_metrics': performance_metrics,
            'parsing_details': parsing_info,
            'additional_info': additional_info,
            'debug_information': debug_info,
            'enhanced_transactions': enhanced_transactions,
            'export_timestamp': datetime.now().isoformat(),
            'parser_version': '2.0.0',
            'recommendations': {
                'fraud_prevention': self._generate_fraud_prevention_recommendations(transactions, risk_assessment),
                'performance_optimization': self._generate_performance_recommendations(performance_metrics),
                'data_quality': self._generate_data_quality_recommendations(transactions)
            }
        }
        
        return advanced_report
    
    def _generate_fraud_prevention_recommendations(self, transactions: List[Dict[str, Any]], 
                                                 risk_assessment: Dict[str, Any]) -> List[str]:
        """Generate fraud prevention recommendations based on analysis."""
        recommendations = []
        
        risk_level = risk_assessment.get('risk_level', 'Low')
        
        if risk_level == 'High':
            recommendations.append('Implement immediate transaction monitoring')
            recommendations.append('Consider account restrictions for high-value transactions')
            recommendations.append('Enable real-time fraud alerts')
            recommendations.append('Review all transactions manually')
        elif risk_level == 'Medium':
            recommendations.append('Increase monitoring frequency')
            recommendations.append('Set up automated alerts for unusual patterns')
            recommendations.append('Review high-value transactions')
        else:
            recommendations.append('Maintain standard monitoring procedures')
            recommendations.append('Review monthly for any changes in patterns')
        
        # Specific recommendations based on transaction patterns
        if any(t.get('amount_category') == 'Very Large (>1L)' for t in transactions):
            recommendations.append('Implement additional verification for large transactions')
        
        if any(t.get('day_of_week') in ['Saturday', 'Sunday'] for t in transactions):
            recommendations.append('Monitor weekend transactions more closely')
        
        return recommendations
    
    def _generate_performance_recommendations(self, performance_metrics: Dict[str, float]) -> List[str]:
        """Generate performance optimization recommendations."""
        recommendations = []
        
        efficiency_score = performance_metrics.get('efficiency_score', 0)
        parsing_time = performance_metrics.get('parsing_time_seconds', 0)
        memory_usage = performance_metrics.get('memory_usage_mb', 0)
        
        if efficiency_score < 0.5:
            recommendations.append('Consider optimizing parsing algorithms')
            recommendations.append('Review regex patterns for efficiency')
        
        if parsing_time > 5.0:
            recommendations.append('Parsing time is high - consider caching mechanisms')
            recommendations.append('Review text preprocessing steps')
        
        if memory_usage > 500:
            recommendations.append('Memory usage is high - consider streaming processing')
            recommendations.append('Implement memory cleanup in parsing loops')
        
        if efficiency_score > 0.8:
            recommendations.append('Performance is excellent - maintain current configuration')
        
        return recommendations
    
    def _generate_data_quality_recommendations(self, transactions: List[Dict[str, Any]]) -> List[str]:
        """Generate data quality improvement recommendations."""
        recommendations = []
        
        # Check for missing data
        missing_dates = sum(1 for t in transactions if not t.get('date'))
        missing_descriptions = sum(1 for t in transactions if not t.get('description'))
        missing_amounts = sum(1 for t in transactions if t.get('amount') is None)
        missing_balances = sum(1 for t in transactions if t.get('balance') is None)
        
        if missing_dates > 0:
            recommendations.append(f'Review {missing_dates} transactions with missing dates')
        
        if missing_descriptions > 0:
            recommendations.append(f'Review {missing_descriptions} transactions with missing descriptions')
        
        if missing_amounts > 0:
            recommendations.append(f'Review {missing_amounts} transactions with missing amounts')
        
        if missing_balances > 0:
            recommendations.append(f'Review {missing_balances} transactions with missing balances')
        
        # Check for data consistency
        if len(transactions) > 1:
            dates = [t.get('date') for t in transactions if t.get('date')]
            if dates:
                try:
                    sorted_dates = sorted(dates)
                    if dates != sorted_dates:
                        recommendations.append('Transaction dates are not in chronological order')
                except:
                    recommendations.append('Date format inconsistencies detected')
        
        if not recommendations:
            recommendations.append('Data quality is excellent - no issues detected')
        
        return recommendations
    
    def _validate_and_fix_balances(self, transactions: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Validate and fix balance sequences in transactions."""
        if not transactions:
            return transactions
        
        fixed_transactions = []
        
        for i, transaction in enumerate(transactions):
            fixed_transaction = transaction.copy()
            
            # If balance is missing, try to estimate from previous transaction
            if fixed_transaction.get('balance') is None and i > 0:
                prev_balance = fixed_transactions[i-1].get('balance')
                current_amount = transaction.get('amount', 0)
                
                if prev_balance is not None:
                    estimated_balance = prev_balance + current_amount
                    fixed_transaction['balance'] = abs(estimated_balance)
                    fixed_transaction['balance_estimated'] = True
                    
                    if self.debug:
                        print(f"   🔧 Estimated missing balance for transaction {i+1}: {abs(estimated_balance)}")
            
            # Validate balance makes sense
            if i > 0 and fixed_transaction.get('balance') is not None:
                prev_balance = fixed_transactions[i-1].get('balance')
                current_amount = transaction.get('amount', 0)
                current_balance = fixed_transaction['balance']
                
                if prev_balance is not None:
                    expected_balance = prev_balance + current_amount
                    balance_diff = abs(current_balance - abs(expected_balance))
                    
                    # Flag large discrepancies (more than 10% or ₹1000)
                    if balance_diff > max(abs(expected_balance) * 0.1, 1000):
                        fixed_transaction['balance_warning'] = f"Large balance discrepancy detected"
                        if self.debug:
                            print(f"   ⚠️  Balance warning: Expected ~{abs(expected_balance):.2f}, found {current_balance}")
            
            fixed_transactions.append(fixed_transaction)
        
        return fixed_transactions


# Convenience function for easy integration
def parse_statement_layout_aware(raw_text: str, pdf_path: str = "", password: Optional[str] = None, debug: bool = False, user_selected_bank: Optional[str] = None) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """Convenience function to parse any bank statement using layout-aware parsing."""
    parser = LayoutAwareParser(pdf_path, password, debug, user_selected_bank)
    return parser.parse_statement(raw_text)

