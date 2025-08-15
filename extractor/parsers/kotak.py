"""Enhanced Kotak Bank statement parser.

This module handles various Kotak bank statement layouts including:
- Standard single column format with (Dr)/(Cr) indicators
- 6-month statements with split headers
- Different column naming variations
"""
import re
import logging
from typing import List, Dict, Any, Optional
import pandas as pd
from datetime import datetime

logger = logging.getLogger(__name__)


def parse_kotak_single_col(df: pd.DataFrame, debug: bool = False) -> List[Dict[str, Any]]:
    """
    Parse Kotak-like tables where debit/credit are in one column with (Dr)/(Cr).
    Enhanced to handle various layouts including 6-month statements.
    """
    transactions: List[Dict[str, Any]] = []
    
    if df is None or df.empty:
        if debug:
            logger.warning("KOTAK PARSER: DataFrame is empty")
        return transactions

    # Enhanced column detection for different layouts
    cols = list(df.columns)
    if debug:
        logger.info(f"KOTAK PARSER: Available columns: {cols}")
    
    # Try multiple column name patterns with priority
    date_col = _find_col_enhanced(cols, ["date", "dt", "transaction date"])
    desc_col = _find_col_enhanced(cols, ["narration", "description", "particulars", "details"])
    ref_col = _find_col_enhanced(cols, ["chq", "ref", "reference", "chq/ref no", "cheque"])
    
    # CRITICAL FIX: Better amount column detection
    amt_col = _find_amount_column_smart(cols)
    
    bal_col = _find_col_enhanced(cols, ["balance", "running balance", "closing balance"])

    if debug:
        logger.info(f"KOTAK PARSER: Detected columns - Date: {date_col}, Desc: {desc_col}, Ref: {ref_col}, Amount: {amt_col}, Balance: {bal_col}")
        # Add detailed column analysis
        for col in cols:
            col_str = str(col).lower().replace('\n', ' ').replace('\r', ' ').strip()
            logger.info(f"KOTAK PARSER: Column '{col}' -> normalized: '{col_str}'")

    # Validate required columns
    if not date_col or not amt_col:
        if debug:
            logger.warning(f"KOTAK PARSER: Required columns not found. Date: {date_col}, Amount: {amt_col}")
        return []

    # Process each row
    for idx, row in df.iterrows():
        try:
            # Extract and clean data
            date_val = row.get(date_col, '')
            desc_val = row.get(desc_col, '') if desc_col else ''
            ref_val = row.get(ref_col, '') if ref_col else ''
            amount_raw = str(row.get(amt_col, '')).replace(',', '').strip()
            balance_raw = str(row.get(bal_col, '')).replace(',', '').strip() if bal_col else ''

            if debug:
                logger.debug(f"KOTAK PARSER: Row {idx} - Date: '{date_val}', Desc: '{desc_val}', Amount: '{amount_raw}'")

            # Skip header rows and empty rows
            if _is_header_row(date_val, desc_val, amount_raw):
                if debug:
                    logger.debug(f"KOTAK PARSER: Skipping header row {idx}")
                continue

            # Parse date
            date = _parse_kotak_date(date_val)
            if not date:
                if debug:
                    logger.debug(f"KOTAK PARSER: Skipping row {idx} - invalid date: {date_val}")
                continue

            # Parse amount with enhanced Dr/Cr detection
            amount = _parse_kotak_amount_enhanced(amount_raw, debug)
            if amount is None:
                if debug:
                    logger.debug(f"KOTAK PARSER: Skipping row {idx} - invalid amount: {amount_raw}")
                continue

            # Parse balance
            balance = _parse_kotak_balance_enhanced(balance_raw, debug)

            # Clean description
            description = _clean_description(desc_val)
            
            # Clean reference
            reference = _clean_reference(ref_val)

            # Create transaction
            transaction = {
                'date': date,
                'description': description,
                'ref': reference,
                'amount': amount,
                'balance': balance,
                'category': _categorize_kotak_transaction(description, amount),
                'bank_type': 'KOTAK',
                'parser_confidence': 'high'
            }

            transactions.append(transaction)
            if debug:
                logger.debug(f"KOTAK PARSER: Added transaction {idx}: {transaction}")

        except Exception as e:
            if debug:
                logger.error(f"KOTAK PARSER: Error processing row {idx}: {e}")
            continue

    if debug:
        logger.info(f"KOTAK PARSER: Extracted {len(transactions)} transactions")

    return transactions


def _find_col_enhanced(columns: List, patterns: List[str]) -> Optional[str]:
    """Enhanced column finder that handles various naming patterns."""
    for col in columns:
        col_str = str(col).lower().replace('\n', ' ').replace('\r', ' ').strip()
        
        # Try exact match first
        if col_str in patterns:
            return col
        
        # Try partial matches
        for pattern in patterns:
            if pattern in col_str:
                return col
            
        # Try pattern variations
        for pattern in patterns:
            # Handle split headers like "Withdrawal(Dr)/" + "Deposit(Cr)"
            if any(part in col_str for part in pattern.split('/')):
                return col
    
    return None


def _find_amount_column_smart(columns: List) -> Optional[str]:
    """Smart amount column detection that avoids conflicts with other columns."""
    # First, identify other important columns to avoid conflicts
    date_patterns = ["date", "dt", "transaction date"]
    desc_patterns = ["narration", "description", "particulars", "details"]
    ref_patterns = ["chq", "ref", "reference", "chq/ref no", "cheque"]
    balance_patterns = ["balance", "running balance", "closing balance"]
    
    # Find these columns first
    date_col = _find_col_enhanced(columns, date_patterns)
    desc_col = _find_col_enhanced(columns, desc_patterns)
    ref_col = _find_col_enhanced(columns, ref_patterns)
    balance_col = _find_col_enhanced(columns, balance_patterns)
    
    # Now look for amount column, avoiding conflicts
    amount_patterns = [
        "withdrawal(dr)/deposit(cr)",  # Full pattern first
        "withdrawal(dr)/",             # Split header part 1
        "deposit(cr)",                 # Split header part 2
        "withdrawal",                  # Partial match
        "deposit",                     # Partial match
        "amount",                      # Generic
    ]
    
    for col in columns:
        col_str = str(col).lower().replace('\n', ' ').replace('\r', ' ').strip()
        
        # Skip if this column is already mapped to something else
        if col == date_col or col == desc_col or col == ref_col or col == balance_col:
            continue
            
        # Check if this column matches amount patterns
        for pattern in amount_patterns:
            if pattern in col_str:
                return col
    
    return None


def _is_header_row(date_val: Any, desc_val: Any, amount_raw: str) -> bool:
    """Check if row is a header row."""
    if pd.isna(date_val) or pd.isna(desc_val):
        return True
    
    date_str = str(date_val).lower().strip()
    desc_str = str(desc_val).lower().strip()
    amount_str = str(amount_raw).lower().strip()
    
    # Skip rows with header-like content
    header_indicators = ['date', 'narration', 'description', 'chq', 'ref', 'withdrawal', 'deposit', 'balance']
    
    if any(indicator in date_str for indicator in header_indicators):
        return True
    if any(indicator in desc_str for indicator in header_indicators):
        return True
    if any(indicator in amount_str for indicator in header_indicators):
        return True
    
    return False


def _parse_kotak_date(date_val: Any) -> Optional[str]:
    """Parse Kotak date formats."""
    if pd.isna(date_val) or not date_val:
        return None
    
    date_str = str(date_val).strip()
    
    # Common Kotak date formats
    date_formats = [
        '%d-%m-%Y', '%d/%m/%Y', '%d-%m-%y', '%d/%m/%y',
        '%d-%m', '%d/%m'  # Just day/month
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
    
    # Try regex patterns
    patterns = [
        r'(\d{1,2})[-/](\d{1,2})[-/](\d{2,4})',  # DD-MM-YYYY
        r'(\d{1,2})[-/](\d{1,2})',  # DD-MM
    ]
    
    for pattern in patterns:
        match = re.match(pattern, date_str)
        if match:
            try:
                if len(match.groups()) == 3:
                    day, month, year = match.groups()
                    day = int(day)
                    month = int(month)
                    year = int(year)
                    
                    if year < 100:
                        year = year + 2000 if year < 50 else year + 1900
                    
                    parsed = datetime(year, month, day)
                    return parsed.strftime('%Y-%m-%d')
                elif len(match.groups()) == 2:
                    day, month = match.groups()
                    day = int(day)
                    month = int(month)
                    
                    # Assume current year for 6-month statements
                    current_year = datetime.now().year
                    parsed = datetime(current_year, month, day)
                    return parsed.strftime('%Y-%m-%d')
            except (ValueError, TypeError):
                continue
    
    return None


def _parse_kotak_amount_enhanced(amount_raw: str, debug: bool = False) -> Optional[float]:
    """Enhanced amount parsing for Kotak statements."""
    if not amount_raw or amount_raw == '':
        return None
    
    try:
        # Remove commas and clean up
        amount_str = amount_raw.replace(',', '').strip()
        
        # Handle Dr/Cr indicators
        if amount_str.endswith('(Dr)'):
            # Debit - negative amount
            amount_value = amount_str[:-4].strip()
            try:
                return -float(amount_value)
            except ValueError:
                if debug:
                    logger.debug(f"KOTAK PARSER: Could not parse Dr amount: {amount_value}")
                return None
                
        elif amount_str.endswith('(Cr)'):
            # Credit - positive amount
            amount_value = amount_str[:-4].strip()
            try:
                return float(amount_value)
            except ValueError:
                if debug:
                    logger.debug(f"KOTAK PARSER: Could not parse Cr amount: {amount_value}")
                return None
        
        else:
            # Try to parse as regular number
            try:
                return float(amount_str)
            except ValueError:
                if debug:
                    logger.debug(f"KOTAK PARSER: Could not parse amount: {amount_str}")
                return None
                
    except Exception as e:
        if debug:
            logger.debug(f"KOTAK PARSER: Amount parsing error: {e}")
        return None


def _parse_kotak_balance_enhanced(balance_raw: str, debug: bool = False) -> Optional[float]:
    """Enhanced balance parsing for Kotak statements."""
    if not balance_raw or balance_raw == '':
        return None
    
    try:
        # Remove commas and clean up
        balance_str = balance_raw.replace(',', '').strip()
        
        # Handle Dr/Cr indicators in balance
        if balance_str.endswith('(Dr)'):
            balance_value = balance_str[:-4].strip()
            try:
                return -float(balance_value)
            except ValueError:
                return None
                
        elif balance_str.endswith('(Cr)'):
            balance_value = balance_str[:-4].strip()
            try:
                return float(balance_value)
            except ValueError:
                return True
        
        else:
            # Try to parse as regular number
            try:
                return float(balance_str)
            except ValueError:
                return None
                
    except Exception:
        return None


def _clean_description(desc_val: Any) -> str:
    """Clean and format description."""
    if pd.isna(desc_val):
        return ''
    
    desc_str = str(desc_val).strip()
    
    # Remove excessive whitespace and newlines
    desc_str = re.sub(r'\s+', ' ', desc_str)
    
    return desc_str if desc_str else 'No Description'


def _clean_reference(ref_val: Any) -> str:
    """Clean and format reference."""
    if pd.isna(ref_val):
        return ''
    
    ref_str = str(ref_val).strip()
    
    # Remove excessive whitespace and newlines
    ref_str = re.sub(r'\s+', ' ', ref_str)
    
    return ref_str if ref_str else ''


def _categorize_kotak_transaction(description: str, amount: float) -> str:
    """Categorize Kotak transaction based on description and amount."""
    desc_lower = description.lower()
    
    # Kotak-specific patterns
    if any(k in desc_lower for k in ['upi', 'unified payment']):
        return 'UPI'
    
    if any(k in desc_lower for k in ['neft', 'imps', 'rtgs']):
        return 'Transfer'
    
    if any(k in desc_lower for k in ['atm', 'cash withdrawal']):
        return 'ATM'
    
    if any(k in desc_lower for k in ['nach', 'mandate']):
        return 'NACH'
    
    if any(k in desc_lower for k in ['irctc', 'railway']):
        return 'Travel'
    
    if any(k in desc_lower for k in ['netflix', 'entertainment']):
        return 'Entertainment'
    
    if any(k in desc_lower for k in ['gas', 'fuel']):
        return 'Fuel'
    
    if any(k in desc_lower for k in ['indmoney', 'investment']):
        return 'Investment'
    
    # Amount-based categorization
    if amount > 0:
        return 'Credit'
    else:
        return 'Debit'


# Legacy function for backward compatibility
def _find_col(columns, patterns):
    """Legacy column finder function."""
    return _find_col_enhanced(columns, patterns)


