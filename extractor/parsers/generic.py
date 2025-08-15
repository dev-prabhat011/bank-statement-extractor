"""Enhanced generic parser for unknown bank statement formats.

This parser uses intelligent pattern recognition to automatically
identify and parse transactions from various table structures.
"""
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd
from collections import defaultdict

logger = logging.getLogger(__name__)


def parse_generic_adaptive(df: pd.DataFrame, debug: bool = False, 
                          template_hints: Optional[Dict] = None) -> List[Dict[str, Any]]:
    """
    Intelligent generic parser that adapts to different table structures.
    
    Args:
        df: DataFrame containing the transaction table
        debug: Enable debug logging
        template_hints: Optional hints from template detection
        
    Returns:
        List of transaction dictionaries
    """
    transactions = []
    
    if df is None or df.empty:
        logger.info("[GENERIC ADAPTIVE PARSER] DataFrame is empty, skipping.")
        return []

    # Clean columns and handle various table formats
    df = _clean_table_columns_advanced(df)
    if df is None or df.empty:
        logger.info("[GENERIC ADAPTIVE PARSER] DataFrame could not be cleaned, skipping.")
        return []

    # Analyze table structure
    table_analysis = _analyze_table_structure(df, debug)
    
    # Identify columns using multiple strategies
    col_map = _identify_columns_intelligent(df, table_analysis, template_hints, debug)
    
    if debug:
        logger.info(f"[GENERIC ADAPTIVE PARSER] Detected columns: {col_map}")
        logger.info(f"[GENERIC ADAPTIVE PARSER] Table analysis: {table_analysis}")
    
    # Validate essential columns
    if not _validate_essential_columns(col_map, debug):
        logger.warning("[GENERIC ADAPTIVE PARSER] Essential columns not found, attempting fallback detection.")
        col_map = _fallback_column_detection(df, debug)
        
        if not _validate_essential_columns(col_map, debug):
            logger.error("[GENERIC ADAPTIVE PARSER] Cannot proceed without essential columns.")
            return []

    # Parse transactions with adaptive strategies
    transactions = _parse_transactions_adaptive(df, col_map, table_analysis, debug)
    
    # Post-process and validate transactions
    transactions = _post_process_transactions(transactions, debug)
    
    if debug:
        logger.info(f"[GENERIC ADAPTIVE PARSER] Extracted {len(transactions)} transactions.")
    
        return transactions


def _clean_table_columns_advanced(df: pd.DataFrame) -> pd.DataFrame:
    """Advanced table column cleaning with multiple strategies."""
    clean_df = df.copy()
    
    # Strategy 1: Handle completely numeric or unnamed columns
    if all(isinstance(col, int) or str(col).startswith('Unnamed:') for col in clean_df.columns):
        if not clean_df.empty:
            first_row_values = [str(val).strip() for val in clean_df.iloc[0]]
            # Check if first row looks like headers
            if any(re.search(r'[a-zA-Z]', h) for h in first_row_values if h):
                logger.info("Using first row as header (columns were unnamed/numeric)")
                clean_df.columns = first_row_values
                clean_df = clean_df.iloc[1:].reset_index(drop=True)
            else:
                # Assign descriptive names based on content analysis
                clean_df.columns = _generate_intelligent_column_names(clean_df)
    
    # Strategy 2: Clean existing string column names
    new_columns = []
    for col in clean_df.columns:
        if pd.isna(col):
            new_name = f"unnamed_{len(new_columns)}"
        else:
            new_name = str(col).strip().replace('\n', ' ').replace('\r', ' ')
            # Standardize common terms
            new_name = _standardize_column_name(new_name)
        new_columns.append(new_name)
    
    clean_df.columns = new_columns
    
    # Strategy 3: Remove completely empty columns and rows
    clean_df.dropna(axis=1, how='all', inplace=True)
    clean_df.dropna(axis=0, how='all', inplace=True)
    
    # Strategy 4: Handle merged cells and split columns if needed
    clean_df = _handle_merged_cells(clean_df)
    
    return clean_df


def _generate_intelligent_column_names(df: pd.DataFrame) -> List[str]:
    """Generate intelligent column names based on content analysis."""
    column_names = []
    
    for i, col in enumerate(df.columns):
        # Sample data from the column
        sample_data = df.iloc[:, i].astype(str).dropna().head(10)
        
        # Analyze content patterns
        if _is_likely_date_column(sample_data):
            column_names.append("Date")
        elif _is_likely_description_column(sample_data):
            column_names.append("Description")
        elif _is_likely_amount_column(sample_data):
            column_names.append("Amount")
        elif _is_likely_balance_column(sample_data):
            column_names.append("Balance")
        elif _is_likely_reference_column(sample_data):
            column_names.append("Reference")
        else:
            column_names.append(f"Column_{i+1}")
    
    return column_names


def _standardize_column_name(col_name: str) -> str:
    """Standardize column names to common terms."""
    col_lower = col_name.lower()
    
    # Date variations
    if any(term in col_lower for term in ['date', 'dt', 'transaction date', 'value date']):
        return 'Date'
    
    # Description variations
    if any(term in col_lower for term in ['description', 'details', 'narration', 'particulars', 'memo', 'remarks']):
        return 'Description'
    
    # Amount variations
    if any(term in col_lower for term in ['amount', 'value', 'sum', 'total']):
        return 'Amount'
    
    # Debit variations
    if any(term in col_lower for term in ['debit', 'withdrawal', 'payment', 'outgoing', 'dr']):
        return 'Debit'
    
    # Credit variations
    if any(term in col_lower for term in ['credit', 'deposit', 'receipt', 'incoming', 'cr']):
        return 'Credit'
    
    # Balance variations
    if any(term in col_lower for term in ['balance', 'running balance', 'acct balance', 'closing balance']):
        return 'Balance'
    
    # Reference variations
    if any(term in col_lower for term in ['reference', 'ref', 'cheque', 'chq', 'transaction id']):
        return 'Reference'
    
    return col_name


def _analyze_table_structure(df: pd.DataFrame, debug: bool = False) -> Dict[str, Any]:
    """Analyze table structure to understand layout and content patterns."""
    analysis = {
        'total_rows': len(df),
        'total_columns': len(df.columns),
        'has_date_column': False,
        'has_description_column': False,
        'has_amount_column': False,
        'has_balance_column': False,
        'has_separate_debit_credit': False,
        'has_withdrawal_deposit': False,
        'has_single_amount_column': False,
        'has_dr_cr_indicators': False,
        'column_types': {},
        'data_patterns': {}
    }
    
    for col_name in df.columns:
        col_data = df[col_name].astype(str).dropna().head(20)
        if col_data.empty:
            continue
            
        # Analyze column content
        col_analysis = _analyze_column_content(col_data, col_name)
        analysis['column_types'][col_name] = col_analysis
        
        # Update overall analysis
        if col_analysis['is_date']:
            analysis['has_date_column'] = True
        if col_analysis['is_description']:
            analysis['has_description_column'] = True
        if col_analysis['is_amount']:
            analysis['has_amount_column'] = True
        if col_analysis['is_balance']:
            analysis['has_balance_column'] = True
        if col_analysis['is_debit']:
            analysis['has_separate_debit_credit'] = True
        if col_analysis['is_credit']:
            analysis['has_separate_debit_credit'] = True
        if col_analysis['is_withdrawal']:
            analysis['has_withdrawal_deposit'] = True
        if col_analysis['is_deposit']:
            analysis['has_withdrawal_deposit'] = True
        if col_analysis['has_dr_cr']:
            analysis['has_dr_cr_indicators'] = True
    
    # Determine amount format
    if analysis['has_separate_debit_credit']:
        analysis['amount_format'] = 'separate_debit_credit'
    elif analysis['has_withdrawal_deposit']:
        analysis['amount_format'] = 'separate_withdrawal_deposit'
    elif analysis['has_amount_column']:
        analysis['amount_format'] = 'single_amount_column'
    else:
        analysis['amount_format'] = 'unknown'
    
    if debug:
        logger.debug(f"Table structure analysis: {analysis}")
    
    return analysis


def _analyze_column_content(col_data: pd.Series, col_name: str) -> Dict[str, Any]:
    """Analyze the content of a single column to determine its type."""
    analysis = {
        'column_name': col_name,
        'is_date': False,
        'is_description': False,
        'is_amount': False,
        'is_balance': False,
        'is_debit': False,
        'is_credit': False,
        'is_withdrawal': False,
        'is_deposit': False,
        'has_dr_cr': False,
        'confidence': 0.0
    }
    
    # Date detection
    date_patterns = [
        r'\d{1,2}[/-]\d{1,2}[/-]\d{2,4}',  # DD/MM/YYYY
        r'\d{1,2}\s+[A-Za-z]{3}',  # DD MMM
        r'[A-Za-z]{3}\s+\d{1,2}',  # MMM DD
    ]
    
    date_matches = sum(col_data.str.contains(pattern, regex=True) for pattern in date_patterns)
    if date_matches >= len(col_data) * 0.3:  # At least 30% look like dates
        analysis['is_date'] = True
        analysis['confidence'] += 0.8
    
    # Description detection
    text_ratio = col_data.str.contains(r'[a-zA-Z]').sum() / len(col_data)
    avg_words = col_data.str.count(r'\s+').mean()
    if text_ratio > 0.7 and avg_words > 1.0:
        analysis['is_description'] = True
        analysis['confidence'] += 0.7
    
    # Amount detection
    amount_patterns = [
        r'^\s*\(?[$€£]?[\d,.-]+\)?\s*$',  # Currency amounts
        r'^\s*[\d,.-]+\s*$',  # Plain numbers
    ]
    
    amount_matches = sum(col_data.str.match(pattern, regex=True) for pattern in amount_patterns)
    if amount_matches >= len(col_data) * 0.5:  # At least 50% look like amounts
        analysis['is_amount'] = True
        analysis['confidence'] += 0.8
        
        # Check for DR/CR indicators
        dr_cr_matches = col_data.str.contains(r'\(?dr\)?|\(?cr\)?', regex=True, case=False).sum()
        if dr_cr_matches > 0:
            analysis['has_dr_cr'] = True
    
    # Balance detection
    if 'balance' in col_name.lower() or analysis['is_amount']:
        analysis['is_balance'] = True
        analysis['confidence'] += 0.6
    
    # Debit/Credit detection
    if 'debit' in col_name.lower() or 'dr' in col_name.lower():
        analysis['is_debit'] = True
        analysis['confidence'] += 0.9
    
    if 'credit' in col_name.lower() or 'cr' in col_name.lower():
        analysis['is_credit'] = True
        analysis['confidence'] += 0.9
    
    # Withdrawal/Deposit detection
    if 'withdrawal' in col_name.lower() or 'payment' in col_name.lower():
        analysis['is_withdrawal'] = True
        analysis['confidence'] += 0.9
    
    if 'deposit' in col_name.lower() or 'receipt' in col_name.lower():
        analysis['is_deposit'] = True
        analysis['confidence'] += 0.9
    
    return analysis


def _identify_columns_intelligent(df: pd.DataFrame, table_analysis: Dict, 
                                 template_hints: Optional[Dict], debug: bool = False) -> Dict[str, str]:
    """Intelligent column identification using multiple strategies."""
    col_mapping = {}
    cols = list(df.columns)
    
    # Strategy 1: Use template hints if available
    if template_hints and template_hints.get('parsing_hints'):
        hints = template_hints['parsing_hints']
        if debug:
            logger.debug(f"Using template hints: {hints}")
        
        # Apply hints to column identification
        if hints.get('amount_parsing'):
            if 'single_column_with_dr_cr' in str(hints['amount_parsing']):
                # Look for single amount column with DR/CR
                for col in cols:
                    if table_analysis['column_types'].get(col, {}).get('has_dr_cr'):
                        col_mapping['amount'] = col
                        break
    
    # Strategy 2: Use table analysis results
    for col_name, col_analysis in table_analysis['column_types'].items():
        if col_analysis['is_date'] and 'date' not in col_mapping:
            col_mapping['date'] = col_name
        elif col_analysis['is_description'] and 'desc' not in col_mapping:
            col_mapping['desc'] = col_name
        elif col_analysis['is_balance'] and 'balance' not in col_mapping:
            col_mapping['balance'] = col_name
        elif col_analysis['is_debit'] and 'debit' not in col_mapping:
            col_mapping['debit'] = col_name
        elif col_analysis['is_credit'] and 'credit' not in col_mapping:
            col_mapping['credit'] = col_name
        elif col_analysis['is_withdrawal'] and 'debit' not in col_mapping:
            col_mapping['debit'] = col_name
        elif col_analysis['is_deposit'] and 'credit' not in col_mapping:
            col_mapping['credit'] = col_name
        elif col_analysis['is_amount'] and 'amount' not in col_mapping:
            col_mapping['amount'] = col_name
    
    # Strategy 3: Fallback to header-based detection
    if not col_mapping.get('date'):
        for col in cols:
            if any(term in str(col).lower() for term in ['date', 'dt']):
                col_mapping['date'] = col
                break
    
    if not col_mapping.get('desc'):
        for col in cols:
            if any(term in str(col).lower() for term in ['description', 'details', 'narration']):
                col_mapping['desc'] = col
                break
    
    # Strategy 4: Content-based detection for remaining columns
    if not col_mapping.get('amount') and not (col_mapping.get('debit') and col_mapping.get('credit')):
        # Look for numeric columns that could be amounts
        for col in cols:
            if col not in col_mapping.values():
                col_analysis = table_analysis['column_types'].get(col, {})
                if col_analysis.get('is_amount'):
                    col_mapping['amount'] = col
                    break
    
    return col_mapping


def _validate_essential_columns(col_map: Dict[str, str], debug: bool = False) -> bool:
    """Validate that essential columns are present."""
    # Must have date
    if 'date' not in col_map:
        if debug:
            logger.warning("Missing date column")
        return False
    
    # Must have either amount OR (debit AND credit)
    has_amount = 'amount' in col_map
    has_debit_credit = 'debit' in col_map and 'credit' in col_map
    
    if not (has_amount or has_debit_credit):
        if debug:
            logger.warning("Missing amount column or debit/credit columns")
        return False
    
    return True


def _fallback_column_detection(df: pd.DataFrame, debug: bool = False) -> Dict[str, str]:
    """Fallback column detection when primary methods fail."""
    col_mapping = {}
    cols = list(df.columns)
    
    # Simple fallback: assume first few columns are date, description, amount
    if len(cols) >= 3:
        col_mapping['date'] = cols[0]
        col_mapping['desc'] = cols[1]
        col_mapping['amount'] = cols[2]
        
        if len(cols) >= 4:
            col_mapping['balance'] = cols[3]
    
    if debug:
        logger.info(f"Fallback column detection: {col_mapping}")
    
    return col_mapping


def _parse_transactions_adaptive(df: pd.DataFrame, col_map: Dict[str, str], 
                                table_analysis: Dict, debug: bool = False) -> List[Dict[str, Any]]:
    """Parse transactions using adaptive strategies based on detected structure."""
    transactions = []
    
    for idx, row in df.iterrows():
        try:
            # Parse date
            date_val = row.get(col_map.get('date', ''), '')
            date = _parse_date_adaptive(date_val, debug)
            if not date:
                continue

            # Parse description
            desc = row.get(col_map.get('desc', ''), '')
            if pd.isna(desc):
                desc = ''
            desc = str(desc).strip()

            # Parse amount based on detected format
            amount = _parse_amount_adaptive(row, col_map, table_analysis, debug)
            if amount is None:
                continue

            # Parse balance
            balance = None
            if col_map.get('balance'):
                balance = _parse_balance_adaptive(row.get(col_map['balance'], ''), debug)

            # Parse reference
            ref = row.get(col_map.get('ref', ''), '')
            if pd.isna(ref):
                ref = ''
            ref = str(ref).strip()

            # Categorize transaction
            category = _categorize_transaction_adaptive(desc, amount, debug)

            transaction = {
                'date': date,
                'description': desc,
                'ref': ref,
                'amount': amount,
                'balance': balance,
                'category': category,
                'bank_type': 'Generic',
                'parser_confidence': 'medium'
            }
            
            transactions.append(transaction)
            
        except Exception as e:
            if debug:
                logger.error(f"Error processing row {idx}: {e}")
            continue

    return transactions


def _parse_date_adaptive(date_val: Any, debug: bool = False) -> Optional[str]:
    """Adaptive date parsing for various formats."""
    if pd.isna(date_val) or not date_val:
        return None
    
    date_str = str(date_val).strip()
    
    # Try multiple date formats
    date_formats = [
        '%Y-%m-%d', '%d/%m/%Y', '%d-%m-%Y', '%Y/%m/%d',
        '%d/%m/%y', '%d-%m-%y', '%m/%d/%Y', '%m-%d-%Y',
        '%d %b %Y', '%d-%b-%Y', '%b %d %Y', '%b-%d-%Y',
        '%d %B %Y', '%d-%B-%Y', '%B %d %Y', '%B-%d-%Y',
        '%d/%m', '%d-%m', '%d %b', '%d-%b'
    ]
    
    for fmt in date_formats:
        try:
            parsed = datetime.strptime(date_str, fmt)
            
            # Handle 2-digit years
            if parsed.year < 100:
                if parsed.year < 50:
                    parsed = parsed.replace(year=parsed.year + 2000)
                else:
                    parsed = parsed.replace(year=parsed.year + 1900)
            
            return parsed.strftime('%Y-%m-%d')
            
        except ValueError:
            continue
    
    return None


def _parse_amount_adaptive(row: pd.Series, col_map: Dict[str, str], 
                          table_analysis: Dict, debug: bool = False) -> Optional[float]:
    """Adaptive amount parsing based on detected structure."""
    try:
        # Check if we have separate debit/credit columns
        if col_map.get('debit') and col_map.get('credit'):
            debit_val = row.get(col_map['debit'], '')
            credit_val = row.get(col_map['credit'], '')
            
            if not pd.isna(debit_val) and str(debit_val).strip():
                debit_str = str(debit_val).replace(',', '').strip()
                if debit_str and debit_str != '0':
                    return -abs(float(debit_str))
            
            if not pd.isna(credit_val) and str(credit_val).strip():
                credit_str = str(credit_val).replace(',', '').strip()
                if credit_str and credit_str != '0':
                    return abs(float(credit_str))
        
        # Check if we have a single amount column
        elif col_map.get('amount'):
            amount_val = row.get(col_map['amount'], '')
            if not pd.isna(amount_val) and str(amount_val).strip():
                amount_str = str(amount_val).replace(',', '').strip()
                
                # Handle DR/CR indicators
                is_negative = False
                if re.search(r'\(?dr\)?', amount_str, re.IGNORECASE):
                    is_negative = True
                    amount_str = re.sub(r'\(?dr\)?', '', amount_str, flags=re.IGNORECASE)
                elif re.search(r'\(?cr\)?', amount_str, re.IGNORECASE):
                    is_negative = False
                    amount_str = re.sub(r'\(?cr\)?', '', amount_str, flags=re.IGNORECASE)
                
                # Handle parentheses for negatives
                if amount_str.startswith('(') and amount_str.endswith(')'):
                    amount_str = amount_str[1:-1]
                    is_negative = True
                
                if amount_str and amount_str != '0':
                    amount = float(amount_str)
                    return -amount if is_negative else amount
        
        return None
        
    except (ValueError, TypeError) as e:
        if debug:
            logger.debug(f"Amount parsing error: {e}")
        return None


def _parse_balance_adaptive(balance_val: Any, debug: bool = False) -> Optional[float]:
    """Adaptive balance parsing."""
    if pd.isna(balance_val) or not balance_val:
        return None
    
    try:
        balance_str = str(balance_val).replace(',', '').strip()
        
        # Handle negative balances
        is_negative = False
        if balance_str.startswith('(') and balance_str.endswith(')'):
            balance_str = balance_str[1:-1]
            is_negative = True
        elif balance_str.endswith('-'):
            balance_str = balance_str[:-1]
            is_negative = True
        elif balance_str.startswith('-'):
            balance_str = balance_str[1:]
            is_negative = True
        
        if balance_str and balance_str != '0':
            amount = float(balance_str)
            return -amount if is_negative else amount
        
        return None
        
    except (ValueError, TypeError) as e:
        if debug:
            logger.debug(f"Balance parsing error: {e}")
        return None


def _categorize_transaction_adaptive(desc: str, amount: float, debug: bool = False) -> str:
    """Adaptive transaction categorization."""
    desc_lower = desc.lower()
    
    # Enhanced categorization patterns
    if any(k in desc_lower for k in ['atm', 'cash withdrawal', 'cash dispensed']):
        return 'ATM'
    
    if any(k in desc_lower for k in ['cheque', 'chq', 'cq no', 'chq no']):
        return 'Cheque'
    
    if any(k in desc_lower for k in ['neft', 'imps', 'rtgs', 'transfer']):
        return 'Transfer'
    
    if any(k in desc_lower for k in ['salary', 'payroll', 'wages', 'remuneration']):
        return 'Salary'
    
    if any(k in desc_lower for k in ['emi', 'loan', 'installment', 'instalment']):
        return 'EMI'
    
    if any(k in desc_lower for k in ['interest', 'int.', 'interest credit']):
        return 'Interest'
    
    if any(k in desc_lower for k in ['fee', 'charge', 'penalty', 'fine', 'service charge']):
        return 'Fee'
    
    if any(k in desc_lower for k in ['refund', 'reversal', 'chargeback', 'adjustment']):
        return 'Refund'
    
    if any(k in desc_lower for k in ['upi', 'gpay', 'phonepe', 'paytm']):
        return 'UPI'
    
    if any(k in desc_lower for k in ['cash deposit', 'cash']):
        return 'Cash'
    
    # Amount-based categorization
    if amount > 0:
        return 'Credit'
    else:
        return 'Debit'


def _post_process_transactions(transactions: List[Dict], debug: bool = False) -> List[Dict]:
    """Post-process transactions for consistency and validation."""
    if not transactions:
        return transactions
    
    # Sort by date
    try:
        transactions.sort(key=lambda x: x.get('date', ''))
    except:
        pass
    
    # Validate and clean data
    cleaned_transactions = []
    for trans in transactions:
        # Ensure required fields
        if not trans.get('date') or trans.get('amount') is None:
            continue
        
        # Clean description
        if trans.get('description'):
            trans['description'] = str(trans['description']).strip()
        
        # Ensure amount is numeric
        try:
            trans['amount'] = float(trans['amount'])
        except (ValueError, TypeError):
            continue
        
        cleaned_transactions.append(trans)
    
    if debug:
        logger.info(f"Post-processing: {len(transactions)} -> {len(cleaned_transactions)} valid transactions")
    
    return cleaned_transactions


def _handle_merged_cells(df: pd.DataFrame) -> pd.DataFrame:
    """Handle potential merged cells in the table."""
    # This is a placeholder for merged cell handling
    # In practice, you might need more sophisticated logic
    return df


# Legacy function for backward compatibility
def _parse_generic(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Legacy generic parser function."""
    return parse_generic_adaptive(df, debug=False)


