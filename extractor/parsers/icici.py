"""ICICI Bank statement parser.

Handles ICICI bank statements with separate debit/credit columns
and various date formats.
"""
import re
import logging
from datetime import datetime
from typing import List, Dict, Any, Optional
import pandas as pd

logger = logging.getLogger(__name__)


def parse_icici_standard(df: pd.DataFrame, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Parse ICICI bank statement transactions.
    
    Args:
        df: DataFrame containing the transaction table
        debug: Enable debug logging
        
    Returns:
        List of transaction dictionaries
    """
    transactions = []
    
    if df is None or df.empty:
        logger.warning("ICICI PARSER: DataFrame is empty")
        return []
    
    # Clean and standardize column names
    df = _clean_icici_columns(df)
    
    # Identify key columns
    col_map = _identify_icici_columns(df)
    
    if debug:
        logger.info(f"ICICI PARSER: Detected columns: {col_map}")
    
    # Validate required columns
    if not col_map.get('date') or not (col_map.get('debit') or col_map.get('credit')):
        logger.warning("ICICI PARSER: Required columns not found")
        return []
    
    # Process each row
    for idx, row in df.iterrows():
        try:
            # Parse date
            date_val = row.get(col_map['date'], '')
            date = _parse_icici_date(date_val)
            if not date:
                if debug:
                    logger.debug(f"ICICI PARSER: Skipping row {idx} - invalid date: {date_val}")
                continue
            
            # Parse description
            desc = row.get(col_map.get('desc', ''), '')
            if pd.isna(desc):
                desc = ''
            desc = str(desc).strip()
            
            # Parse amount
            amount = _parse_icici_amount(
                row.get(col_map.get('debit', ''), ''),
                row.get(col_map.get('credit', ''), ''),
                debug
            )
            if amount is None:
                if debug:
                    logger.debug(f"ICICI PARSER: Skipping row {idx} - invalid amount")
                continue
            
            # Parse balance
            balance = None
            if col_map.get('balance'):
                balance = _parse_icici_balance(row.get(col_map['balance'], ''), debug)
            
            # Parse reference
            ref = row.get(col_map.get('ref', ''), '')
            if pd.isna(ref):
                ref = ''
            ref = str(ref).strip()
            
            # Categorize transaction
            category = _categorize_icici_transaction(desc, amount)
            
            transaction = {
                'date': date,
                'description': desc,
                'ref': ref,
                'amount': amount,
                'balance': balance,
                'category': category,
                'bank_type': 'ICICI',
                'parser_confidence': 'high'
            }
            
            transactions.append(transaction)
            
        except Exception as e:
            if debug:
                logger.error(f"ICICI PARSER: Error processing row {idx}: {e}")
            continue
    
    if debug:
        logger.info(f"ICICI PARSER: Extracted {len(transactions)} transactions")
    
    return transactions


def _clean_icici_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize ICICI column names."""
    clean_df = df.copy()
    
    # Standardize common ICICI column names
    column_mapping = {
        'Transaction Date': 'date',
        'Value Date': 'date',
        'Cheque Number': 'ref',
        'Transaction Remarks': 'desc',
        'Transaction Details': 'desc',
        'Particulars': 'desc',
        'Narration': 'desc',
        'Withdrawal Amt.': 'debit',
        'Deposit Amt.': 'credit',
        'Debit': 'debit',
        'Credit': 'credit',
        'Amount': 'amount',
        'Balance': 'balance',
        'Running Balance': 'balance',
        'Account Balance': 'balance'
    }
    
    # Apply column mapping
    new_columns = []
    for col in clean_df.columns:
        col_str = str(col).strip()
        new_name = column_mapping.get(col_str, col_str)
        new_columns.append(new_name)
    
    clean_df.columns = new_columns
    
    # Remove completely empty columns
    clean_df.dropna(axis=1, how='all', inplace=True)
    
    return clean_df


def _identify_icici_columns(df: pd.DataFrame) -> Dict[str, str]:
    """Identify ICICI-specific columns based on content and headers."""
    col_mapping = {}
    cols = list(df.columns)
    
    # Look for date column
    date_candidates = []
    for col in cols:
        col_lower = str(col).lower()
        if any(term in col_lower for term in ['date', 'dt', 'value']):
            date_candidates.append(col)
    
    if date_candidates:
        col_mapping['date'] = date_candidates[0]
    
    # Look for description column
    desc_candidates = []
    for col in cols:
        col_lower = str(col).lower()
        if any(term in col_lower for term in ['description', 'details', 'remarks', 'particulars', 'narration']):
            desc_candidates.append(col)
    
    if desc_candidates:
        col_mapping['desc'] = desc_candidates[0]
    
    # Look for debit/credit columns
    debit_col = None
    credit_col = None
    
    for col in cols:
        col_lower = str(col).lower()
        if any(term in col_lower for term in ['debit', 'withdrawal', 'payment']):
            debit_col = col
        elif any(term in col_lower for term in ['credit', 'deposit', 'receipt']):
            credit_col = col
    
    if debit_col:
        col_mapping['debit'] = debit_col
    if credit_col:
        col_mapping['credit'] = credit_col
    
    # Look for balance column
    for col in cols:
        col_lower = str(col).lower()
        if any(term in col_lower for term in ['balance', 'running']):
            col_mapping['balance'] = col
            break
    
    # Look for reference column
    for col in cols:
        col_lower = str(col).lower()
        if any(term in col_lower for term in ['cheque', 'ref', 'reference', 'chq']):
            col_mapping['ref'] = col
            break
    
    return col_mapping


def _parse_icici_date(date_val: Any) -> Optional[str]:
    """Parse ICICI date formats."""
    if pd.isna(date_val) or not date_val:
        return None
    
    date_str = str(date_val).strip()
    
    # Try common ICICI date formats
    date_formats = [
        '%d/%m/%Y', '%d-%m-%Y', '%d/%m/%y', '%d-%m-%y',
        '%d/%m', '%d-%m',  # Just day/month
        '%d %b %Y', '%d-%b-%Y',  # 15 Jan 2024
        '%b %d %Y', '%b-%d-%Y'   # Jan 15 2024
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
    
    # Try regex patterns for common formats
    patterns = [
        r'(\d{1,2})[/-](\d{1,2})[/-](\d{2,4})',  # DD/MM/YYYY
        r'(\d{1,2})\s+([A-Za-z]{3})\s+(\d{2,4})',  # DD MMM YYYY
    ]
    
    for pattern in patterns:
        match = re.match(pattern, date_str)
        if match:
            try:
                if len(match.groups()) == 3:
                    day, month, year = match.groups()
                    day = int(day)
                    month = int(month) if month.isdigit() else datetime.strptime(month, '%b').month
                    year = int(year)
                    
                    if year < 100:
                        year = year + 2000 if year < 50 else year + 1900
                    
                    parsed = datetime(year, month, day)
                    return parsed.strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                continue
    
    return None


def _parse_icici_amount(debit_val: Any, credit_val: Any, debug: bool = False) -> Optional[float]:
    """Parse ICICI amount from debit/credit columns."""
    try:
        # Check debit column
        if not pd.isna(debit_val) and str(debit_val).strip():
            debit_str = str(debit_val).replace(',', '').strip()
            if debit_str and debit_str != '0':
                amount = float(debit_str)
                return -abs(amount)  # Debit is negative
        
        # Check credit column
        if not pd.isna(credit_val) and str(credit_val).strip():
            credit_str = str(credit_val).replace(',', '').strip()
            if credit_str and credit_str != '0':
                amount = float(credit_str)
                return abs(amount)  # Credit is positive
        
        # If both are empty or zero, return None
        return None
        
    except (ValueError, TypeError) as e:
        if debug:
            logger.debug(f"ICICI PARSER: Amount parsing error: {e}")
        return None


def _parse_icici_balance(balance_val: Any, debug: bool = False) -> Optional[float]:
    """Parse ICICI balance value."""
    if pd.isna(balance_val) or not balance_val:
        return None
    
    try:
        balance_str = str(balance_val).replace(',', '').strip()
        
        # Handle negative balances (parentheses or minus)
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
            logger.debug(f"ICICI PARSER: Balance parsing error: {e}")
        return None


def _categorize_icici_transaction(desc: str, amount: float) -> str:
    """Categorize ICICI transaction based on description and amount."""
    desc_lower = desc.lower()
    
    # ICICI-specific patterns
    if any(k in desc_lower for k in ['icici', 'bank', 'branch']):
        return 'Bank Charges'
    
    if any(k in desc_lower for k in ['atm', 'cash withdrawal']):
        return 'ATM'
    
    if any(k in desc_lower for k in ['cheque', 'chq', 'cq no']):
        return 'Cheque'
    
    if any(k in desc_lower for k in ['neft', 'imps', 'rtgs']):
        return 'Transfer'
    
    if any(k in desc_lower for k in ['salary', 'payroll', 'wages']):
        return 'Salary'
    
    if any(k in desc_lower for k in ['emi', 'loan', 'installment']):
        return 'EMI'
    
    if any(k in desc_lower for k in ['interest', 'int.']):
        return 'Interest'
    
    if any(k in desc_lower for k in ['fee', 'charge', 'penalty']):
        return 'Fee'
    
    if any(k in desc_lower for k in ['refund', 'reversal']):
        return 'Refund'
    
    # Amount-based categorization
    if amount > 0:
        return 'Credit'
    else:
        return 'Debit'
