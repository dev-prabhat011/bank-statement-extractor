from typing import List, Dict, Any


def parse_hdfc_separate_cols(df) -> List[Dict[str, Any]]:
    """Parse HDFC format with separate DEBIT/CREDIT columns."""
    transactions: List[Dict[str, Any]] = []
    if df is None or df.empty:
        return transactions

    for _, row in df.iterrows():
        date = row.get('DATE')
        desc = row.get('TRANSACTION DETAILS')
        ref = row.get('CHEQUE/REFERENCE#')
        debit = row.get('DEBIT')
        credit = row.get('CREDIT')
        balance = row.get('BALANCE')

        amount = None
        try:
            if debit and str(debit).strip():
                amount = -float(str(debit).replace(',', '').strip())
            elif credit and str(credit).strip():
                amount = float(str(credit).replace(',', '').strip())
        except Exception:
            amount = None

        try:
            balance = float(str(balance).replace(',', '').strip())
        except Exception:
            balance = None

        transactions.append({
            'date': date,
            'description': desc,
            'ref': ref,
            'amount': amount,
            'balance': balance
        })

    return transactions


