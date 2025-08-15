"""High-level orchestrator for bank statement extraction.

This module provides a clean API that uses the modularized components
for analysis, export, and other functionality.
"""
from typing import Tuple, List, Dict, Any, Optional
from . import analysis, exporters
from .parsers.router import parse_with_smart_router


class StatementExtractor:
    """Main extractor class that orchestrates all extraction functionality."""

    def __init__(self,
                 pdf_path: str,
                 start_date_str: str = None,
                 end_date_str: str = None,
                 password: str = None,
                 debug: bool = False,
                 bank_name: str = None,
                 force_unified_parser: bool = True):  # Changed default to True for more robust parsing
        # Import here to avoid circular imports during refactor
        from bank_statement_extractor import BankStatementExtractor
        self._impl = BankStatementExtractor(
            pdf_path,
            start_date_str=start_date_str,
            end_date_str=end_date_str,
            password=password,
            debug=debug,
            bank_name=bank_name
        )
        
        # Store debug flag for enhanced parsing
        self._debug = debug
        self._force_unified_parser = force_unified_parser

    def extract_all(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any], Dict[str, Any]]:
        """Extract all information and perform analysis using modular components."""
        # Always attempt enhanced parsing first for robustness and self-detection
        try:
            # First, let the legacy system decrypt and extract text
            raw_text = self._impl.extract_text()
            
            # Only proceed with enhanced extraction if text was successfully extracted
            raw_text = self._impl.extract_text()
            if raw_text and len(raw_text) > 100:
                print("🔍 Using Enhanced Parser...")
                print(f"📄 Text extracted: {len(raw_text)} characters")
                
                # Now use enhanced extraction for better compatibility
                transactions, parsing_info = self.extract_transactions_enhanced()
                
                print(f"💳 Transactions extracted: {len(transactions)}")
                print(f"📊 Parsing info: {parsing_info}")
                
                # Update the legacy implementation with enhanced results
                self._impl.transactions = transactions
                
                # CRITICAL FIX: Populate high value transactions
                self._impl.high_value_transactions = [
                    t for t in transactions
                    if t.get('amount') is not None and abs(t.get('amount', 0)) >= self._impl.high_value_threshold
                ]
                print(f"💰 High value transactions: {len(self._impl.high_value_transactions)}")
                
                # Extract account info using the extracted text
                self._impl.extract_account_info(raw_text)
                
                # Use modular analysis for enhanced functionality
                analysis.perform_full_analysis(self._impl)
                
                if self._debug:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Enhanced extraction completed: {len(transactions)} transactions")
                    logger.info(f"Parsing info: {parsing_info}")
                    logger.info(f"High value transactions: {len(self._impl.high_value_transactions)}")
                
                return self._impl.transactions, self._impl.account_info, self._impl.analysis_results
            else:
                print("⚠️ Text extraction failed, falling back to legacy parser")
                print(f"📄 Raw text length: {len(raw_text) if raw_text else 'None'}")
                
        except Exception as e:
            if self._debug:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Enhanced extraction failed, falling back to legacy: {e}")
            # Fall back to legacy method
        
        # Use the legacy class for core extraction
        print("🔧 Using Legacy Parser...")
        transactions, account_info, analysis_results = self._impl.extract_all()
        
        # Use modular analysis for enhanced functionality
        analysis.perform_full_analysis(self._impl)
        
        return self._impl.transactions, self._impl.account_info, self._impl.analysis_results

    def extract_transactions_enhanced(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Enhanced transaction extraction using the new unified parsing system.
        
        Returns:
            Tuple of (transactions, parsing_info)
        """
        try:
            from .unified_parser import UnifiedStatementParser
            
            if self._debug:
                print("🚀 Using Unified Parser System...")
            
            # Create unified parser with same settings
            unified_parser = UnifiedStatementParser(
                pdf_path=self._impl.pdf_path,
                password=getattr(self._impl, 'password', None),
                debug=self._debug
            )
            
            # Parse using unified system
            transactions, parsing_info = unified_parser.parse_statement()
            
            if self._debug:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Unified parser extracted {len(transactions)} transactions")
                logger.info(f"Methods tried: {parsing_info.get('total_methods_tried', 0)}")
                logger.info(f"Successful methods: {parsing_info.get('successful_methods', [])}")
                logger.info(f"Bank detected: {parsing_info.get('bank_detected', 'unknown')}")
                logger.info(f"Confidence score: {parsing_info.get('confidence_score', 0.0):.2f}")
            
            return transactions, parsing_info
                
        except Exception as e:
            if self._debug:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Unified parser failed: {e}")
            
            # Fallback to legacy method if unified parser fails
            print("⚠️ Unified parser failed, falling back to legacy method...")
            return self._extract_transactions_legacy_fallback()
    
    def _extract_transactions_legacy_fallback(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Legacy fallback method for when unified parser fails.
        This maintains compatibility with existing functionality.
        """
        try:
            # Use the legacy system's proven table extraction method
            if hasattr(self._impl, 'extract_transactions'):
                print("🔍 Using legacy extract_transactions method...")
                
                # Call the legacy extract_transactions method
                self._impl.extract_transactions()
                
                # Get the extracted transactions
                all_transactions = getattr(self._impl, 'transactions', [])
                
                # If legacy method failed, try direct text parsing (now generalized)
                bank = self._detect_bank()
                if not all_transactions and bank:
                    print(f"🔄 Legacy method failed, trying direct text parsing for {bank}...")
                    all_transactions = self._parse_from_text(bank)
                    parser_used = f'direct_text_parser_{bank}' if all_transactions else 'legacy_parser'
                else:
                    parser_used = 'legacy_parser'
                
                parsing_info = {
                    'total_tables_processed': 1,
                    'successful_tables': 1 if all_transactions else 0,
                    'failed_tables': 0 if all_transactions else 1,
                    'template_detected': 'AUTO_DETECTED',
                    'parser_used': parser_used,
                    'transactions_found': len(all_transactions),
                    'is_legacy_fallback': True,
                    'bank_detected': bank
                }
                
                if self._debug:
                    import logging
                    logger = logging.getLogger(__name__)
                    logger.info(f"Legacy fallback extracted {len(all_transactions)} transactions")
                    logger.info(f"Parsing info: {parsing_info}")
                
                return all_transactions, parsing_info
            else:
                print("⚠️ Legacy extract_transactions method not found...")
                return [], {'error': 'No extraction method available', 'is_legacy_fallback': True}
                
        except Exception as e:
            print(f"❌ Legacy fallback failed: {e}")
            if self._debug:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Legacy fallback failed: {e}")
            return [], {'error': str(e), 'is_legacy_fallback': True}

    def _extract_transactions_enhanced_old(self) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
        """
        Old enhanced transaction extraction using smart parser router.
        This is kept as a fallback.
        """
        try:
            # Extract first page text for template detection
            first_page_text = self._extract_first_page_text_enhanced()
            
            # Get tables from PDF
            tables = self._extract_tables_enhanced()
            
            all_transactions = []
            parsing_info = {
                'total_tables_processed': len(tables),
                'successful_tables': 0,
                'failed_tables': 0,
                'template_detection': {},
                'parser_performance': {}
            }
            
            for i, df in enumerate(tables):
                if df is not None and not df.empty:
                    try:
                        # Use smart parser router
                        transactions, table_parsing_info = parse_with_smart_router(
                            df, first_page_text, debug=self._debug
                        )
                        
                        if transactions:
                            all_transactions.extend(transactions)
                            parsing_info['successful_tables'] += 1
                            
                            # Store template detection info
                            template_key = table_parsing_info.get('template_detected', 'unknown')
                            if template_key not in parsing_info['template_detection']:
                                parsing_info['template_detection'][template_key] = 0
                            parsing_info['template_detection'][template_key] += 1
                            
                            # Store parser performance
                            parser_used = table_parsing_info.get('parser_used', 'unknown')
                            if parser_used not in parsing_info['parser_performance']:
                                parsing_info['parser_performance'][parser_used] = 0
                            parsing_info['parser_performance'][parser_used] += 1
                            
                            if self._debug:
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.info(f"Table {i+1}: {len(transactions)} transactions using {parser_used}")
                        else:
                            parsing_info['failed_tables'] += 1
                            if self._debug:
                                import logging
                                logger = logging.getLogger(__name__)
                                logger.warning(f"Table {i+1}: No transactions extracted")
                                
                    except Exception as e:
                        parsing_info['failed_tables'] += 1
                        if self._debug:
                            import logging
                            logger = logging.getLogger(__name__)
                            logger.error(f"Table {i+1}: Error during parsing: {e}")
                        continue
            
            # Update the legacy implementation with enhanced results
            self._impl.transactions = all_transactions
            
            # Populate high value transactions
            self._impl.high_value_transactions = [
                t for t in all_transactions
                if t.get('amount') is not None and abs(t.get('amount', 0)) >= self._impl.high_value_threshold
            ]
            
            if self._debug:
                import logging
                logger = logging.getLogger(__name__)
                logger.info(f"Enhanced extraction complete: {len(all_transactions)} transactions from {parsing_info['successful_tables']} tables")
                logger.info(f"Template detection: {parsing_info['template_detection']}")
                logger.info(f"Parser performance: {parsing_info['parser_performance']}")
            
            return all_transactions, parsing_info
            
        except Exception as e:
            if self._debug:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Old enhanced transaction extraction failed: {e}")
            # Fallback to legacy method
            return self._impl.transactions, {'error': str(e), 'fallback_used': True}

    def _parse_from_text(self, bank: str) -> List[Dict[str, Any]]:
        """
        Generalized parse transactions directly from extracted text when tabula fails.
        This handles different bank formats with self-detection via bank param.
        """
        try:
            # Get the extracted text
            raw_text = self._impl.extract_text()
            if not raw_text:
                print("❌ No text available for direct parsing")
                return []
            
            print(f"🔍 Parsing {bank} transactions from {len(raw_text)} characters of text...")
            
            # Parse transactions using bank-specific regex patterns
            transactions = []
            skipped_lines = 0
            parsed_lines = 0
            
            # Split text into lines and process each line
            lines = raw_text.split('\n')
            print(f"📄 Processing {len(lines)} lines of text...")
            
            for i, line in enumerate(lines):
                line = line.strip()
                if not line or len(line) < 20:  # Skip very short lines
                    continue
                
                # Try to match transaction pattern for the detected bank
                transaction = self._parse_transaction_line(bank, line)
                if transaction:
                    transactions.append(transaction)
                    parsed_lines += 1
                    if parsed_lines <= 5:  # Show first 5 parsed transactions
                        print(f"   ✅ Line {i+1}: {transaction['date']} - {transaction['description'][:30]}... - {transaction['amount']}")
                else:
                    # Only show skipped lines that look like they might be transactions
                    if any(keyword in line.lower() for keyword in ['upi', 'nach', 'neft', 'imps', 'atm']):
                        skipped_lines += 1
                        if skipped_lines <= 5:  # Show first 5 skipped lines
                            print(f"   ❌ Line {i+1}: Skipped - {line[:50]}...")
            
            print(f"✅ Direct text parsing extracted {len(transactions)} transactions")
            print(f"📊 Summary: {parsed_lines} parsed, {skipped_lines} skipped")
            
            # Show some sample amounts for debugging
            if transactions:
                amounts = [abs(t.get('amount', 0)) for t in transactions if t.get('amount') is not None]
                if amounts:
                    print(f"💰 Amount range: {min(amounts):.2f} to {max(amounts):.2f}")
                    high_value_count = len([amt for amt in amounts if amt >= 1000])
                    print(f"💰 High value transactions (≥1000): {high_value_count}")
            
            return transactions
            
        except Exception as e:
            print(f"❌ Direct text parsing failed: {e}")
            return []

    def _parse_transaction_line(self, bank: str, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single transaction line based on bank-specific patterns.
        """
        try:
            # Skip header lines
            if any(header in line.lower() for header in ['date', 'narration', 'withdrawal', 'deposit', 'balance', 'page']):
                return None
            
            # Bank-specific patterns
            if bank.upper() == 'KOTAK':
                # Pattern for Kotak (adjusted for common formats)
                pattern = r'(\d{2}-\d{2}-\d{4})\s+(.*?)\s+(\d{1,3}(?:,\d{3})*\.?\d*)\s*\((Dr|Cr)\)\s+(\d{1,3}(?:,\d{3})*\.?\d*)\s*\((Dr|Cr)\)'
                match = re.search(pattern, line)
                if match:
                    date_str, description, amount_str, amount_type, balance_str, balance_type = match.groups()
                    
                    # Parse amount
                    try:
                        amount = float(amount_str.replace(',', ''))
                        if amount_type == 'Dr':
                            amount = -amount
                    except ValueError:
                        return None
                    
                    # Parse balance
                    try:
                        balance = float(balance_str.replace(',', ''))
                        if balance_type == 'Dr':
                            balance = -balance
                    except ValueError:
                        balance = None
                    
                    # Parse date
                    try:
                        from datetime import datetime
                        date_obj = datetime.strptime(date_str, '%d-%m-%Y')
                        date = date_obj.strftime('%Y-%m-%d')
                    except ValueError:
                        date = date_str
                    
                    # Clean description
                    description = description.strip()
                    if len(description) > 100:  # Truncate very long descriptions
                        description = description[:97] + "..."
                    
                    return {
                        'date': date,
                        'description': description,
                        'ref': '',  # No reference number in this format
                        'amount': amount,
                        'balance': balance,
                        'category': self._categorize_transaction(description, amount),
                        'bank_type': bank.upper(),
                        'parser_confidence': 'high'
                    }
            
            elif bank.upper() == 'HDFC':
                # Example pattern for HDFC (adjust based on actual HDFC format, e.g., Date Description Debit Credit Balance)
                pattern = r'(\d{2}/\d{2}/\d{2})\s+(.*?)\s+(\d{1,3}(?:,\d{3})*\.?\d*)\s+(\d{1,3}(?:,\d{3})*\.?\d*)\s+(\d{1,3}(?:,\d{3})*\.?\d*)'
                match = re.search(pattern, line)
                if match:
                    date_str, description, debit_str, credit_str, balance_str = match.groups()
                    
                    # Parse amount (debit or credit)
                    try:
                        debit = float(debit_str.replace(',', '')) if debit_str else 0.0
                        credit = float(credit_str.replace(',', '')) if credit_str else 0.0
                        amount = credit - debit
                    except ValueError:
                        return None
                    
                    # Parse balance
                    try:
                        balance = float(balance_str.replace(',', ''))
                    except ValueError:
                        balance = None
                    
                    # Parse date (assuming dd/mm/yy, add year logic if needed)
                    try:
                        from datetime import datetime
                        date_obj = datetime.strptime(date_str, '%d/%m/%y')
                        date = date_obj.strftime('%Y-%m-%d')  # Assume current year or handle rollover
                    except ValueError:
                        date = date_str
                    
                    description = description.strip()
                    
                    return {
                        'date': date,
                        'description': description,
                        'ref': '',
                        'amount': amount,
                        'balance': balance,
                        'category': self._categorize_transaction(description, amount),
                        'bank_type': bank.upper(),
                        'parser_confidence': 'medium'
                    }
            
            elif bank.upper() == 'SBI':
                # Example pattern for SBI (adjust based on actual SBI format, e.g., Txn Date Value Date Description Debit Credit Balance)
                pattern = r'(\d{2}/\d{2}/\d{4})\s+(\d{2}/\d{2}/\d{4})\s+(.*?)\s+(\d{1,3}(?:,\d{3})*\.?\d*)\s+(\d{1,3}(?:,\d{3})*\.?\d*)\s+(-?\d{1,3}(?:,\d{3})*\.?\d*)'
                match = re.search(pattern, line)
                if match:
                    txn_date_str, value_date_str, description, debit_str, credit_str, balance_str = match.groups()
                    
                    # Parse amount
                    try:
                        debit = float(debit_str.replace(',', '')) if debit_str else 0.0
                        credit = float(credit_str.replace(',', '')) if credit_str else 0.0
                        amount = credit - debit
                    except ValueError:
                        return None
                    
                    # Parse balance
                    try:
                        balance = float(balance_str.replace(',', ''))
                    except ValueError:
                        balance = None
                    
                    # Parse date (use txn date)
                    try:
                        from datetime import datetime
                        date_obj = datetime.strptime(txn_date_str, '%d/%m/%Y')
                        date = date_obj.strftime('%Y-%m-%d')
                    except ValueError:
                        date = txn_date_str
                    
                    description = description.strip()
                    
                    return {
                        'date': date,
                        'description': description,
                        'ref': '',
                        'amount': amount,
                        'balance': balance,
                        'category': self._categorize_transaction(description, amount),
                        'bank_type': bank.upper(),
                        'parser_confidence': 'medium'
                    }
            
            # Add patterns for more banks as needed
            
            return None
            
        except Exception as e:
            print(f"❌ Error parsing line '{line[:50]}...': {e}")
            return None

    def _detect_bank(self) -> str:
        """Detect bank from first page text."""
        try:
            first_page_text = self._extract_first_page_text_enhanced().lower()
            if 'kotak' in first_page_text:
                return 'KOTAK'
            if 'hdfc' in first_page_text:
                return 'HDFC'
            if 'sbi' in first_page_text or 'state bank of india' in first_page_text:
                return 'SBI'
            # Add more detections
            return 'UNKNOWN'
        except Exception:
            return 'UNKNOWN'

    def _categorize_transaction(self, description: str, amount: float) -> str:
        """Categorize transaction based on description and amount."""
        desc_lower = description.lower()
        
        # Special handling for NACH
        if 'nach' in desc_lower:
            if any(k in desc_lower for k in ['loan', 'emi', 'instalment', 'installment']):
                return 'EMI'
            if any(k in desc_lower for k in ['sip', 'mutual fund', 'investment', 'pms', 'insurance']):
                return 'Investment'
            return 'NACH'
        
        # Other categories
        if any(k in desc_lower for k in ['upi', 'gpay', 'google pay', 'phonepe', 'paytm']):
            return 'UPI'
        if any(k in desc_lower for k in ['neft']):
            return 'NEFT'
        if any(k in desc_lower for k in ['imps']):
            return 'IMPS'
        if any(k in desc_lower for k in ['atm', 'cash withdrawal']):
            return 'ATM'
        if any(k in desc_lower for k in ['cheque', 'chq', 'cq no', 'chq no']):
            return 'Cheque'
        if any(k in desc_lower for k in ['debit card', 'credit card', 'pos', 'card']):
            return 'Card'
        if any(k in desc_lower for k in ['salary', 'payroll', 'wages', 'remuneration', 'stipend']):
            return 'Salary'
        if any(k in desc_lower for k in ['emi', 'loan', 'instalment', 'installment']):
            return 'EMI'
        if any(k in desc_lower for k in ['interest', 'int.']):
            return 'Interest'
        if any(k in desc_lower for k in ['fee', 'charge', 'penalty', 'fine']):
            return 'Fee'
        if any(k in desc_lower for k in ['refund', 'reversal', 'chargeback']):
            return 'Refund'
        if any(k in desc_lower for k in ['transfer', 'to ', 'from ']):
            return 'Transfer'
        if any(k in desc_lower for k in ['cash deposit', 'cash']):
            return 'Cash'
        if amount > 0 and any(k in desc_lower for k in ['reversal', 'refund']):
            return 'Refund'
        
        return 'Other'

    def _extract_first_page_text_enhanced(self) -> str:
        """Extract first page text with enhanced error handling."""
        # Use the legacy system's extract_text method
        try:
            raw_text = self._impl.extract_text()
            if raw_text:
                # Get first page text from the extracted text
                lines = raw_text.split('\n')
                first_page_lines = []
                char_count = 0
                
                for line in lines:
                    first_page_lines.append(line)
                    char_count += len(line)
                    # Stop when we have enough text (typical first page)
                    if char_count > 2000:
                        break
                
                return '\n'.join(first_page_lines)
        except Exception as e:
            if self._debug:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Enhanced first page extraction failed: {e}")
        
        # Fallback to direct PDF extraction
        try:
            import pdfplumber
            with pdfplumber.open(self._impl.pdf_path) as pdf:
                if pdf.pages:
                    return pdf.pages[0].extract_text() or ''
        except Exception as e:
            if self._debug:
                import logging
                logger = logging.getLogger(__name__)
                logger.warning(f"Enhanced first page extraction failed: {e}")
            # Fallback to legacy method
            return self._extract_first_page_text_legacy()
        return ''

    def _extract_first_page_text_legacy(self) -> str:
        """Legacy first page text extraction."""
        try:
            import PyPDF2
            with open(self._impl.pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                if reader.pages:
                    return reader.pages[0].extract_text() or ''
        except Exception as e:
            if self._debug:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Legacy first page extraction also failed: {e}")
        return ''

    def _extract_tables_enhanced(self) -> List:
        """Extract tables with enhanced options, trying lattice then stream."""
        try:
            from tabula.io import read_pdf
            
            # Use the already-decrypted PDF path
            pdf_path = self._impl.pdf_path
            
            # Try lattice first
            tables = read_pdf(
                pdf_path,
                pages="all",
                multiple_tables=True,
                guess=True,
                lattice=True,
                stream=False,
                password=getattr(self._impl, 'password', None)
            )
            
            # If no tables or empty, try stream mode
            if not tables or all(df.empty for df in tables if df is not None):
                print("⚠️ Lattice mode failed, trying stream mode...")
                tables = read_pdf(
                    pdf_path,
                    pages="all",
                    multiple_tables=True,
                    guess=True,
                    lattice=False,
                    stream=True,
                    password=getattr(self._impl, 'password', None)
                )
            
            return tables if tables else []
            
        except Exception as e:
            if self._debug:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Enhanced table extraction failed: {e}")
            # Fallback to legacy method
            return self._extract_tables_legacy()

    def _extract_tables_legacy(self) -> List:
        """Legacy table extraction method."""
        try:
            from tabula.io import read_pdf
            
            tabula_options = {
                "pages": "all",
                "multiple_tables": True,
                "guess": True,
                "lattice": True,
                "stream": True,
                "password": getattr(self._impl, 'password', None)
            }
            
            tables = read_pdf(self._impl.pdf_path, **tabula_options)
            return tables if tables else []
            
        except Exception as e:
            if self._debug:
                import logging
                logger = logging.getLogger(__name__)
                logger.error(f"Legacy table extraction also failed: {e}")
            return []

    def export_to_excel(self, output_path: str) -> None:
        """Export to Excel using modular export functionality."""
        return exporters.export_excel(self._impl, output_path)

    def to_xml(self, output_path: str = None) -> str:
        """Export to XML using modular export functionality."""
        return exporters.export_xml(self._impl, output_path)

    @property
    def unique_id(self) -> str:
        return getattr(self._impl, 'unique_id', '')

    @property
    def transactions(self) -> List[Dict[str, Any]]:
        return getattr(self._impl, 'transactions', [])

    @property
    def account_info(self) -> Dict[str, Any]:
        return getattr(self._impl, 'account_info', {})

    @property
    def analysis_results(self) -> Dict[str, Any]:
        return getattr(self._impl, 'analysis_results', {})

    @property
    def high_value_transactions(self) -> List[Dict[str, Any]]:
        return getattr(self._impl, 'high_value_transactions', [])

    @property
    def high_value_threshold(self) -> float:
        return getattr(self._impl, 'high_value_threshold', 1000.0)

    @property
    def pdf_path(self) -> str:
        return getattr(self._impl, 'pdf_path', '')

    @property
    def start_date(self) -> Any:
        return getattr(self._impl, 'start_date', None)

    @property
    def debug(self) -> bool:
        return self._debug

    def _should_use_enhanced_parser(self) -> bool:
        """Determine if we should use the enhanced parser."""
        # Always use enhanced parser for robustness, unless explicitly disabled
        if not self._force_unified_parser:
            return False
        
        # Additional checks if needed
        return True