"""Analysis module: Contains all transaction analysis logic.

This module handles recurring debit analysis, salary identification, and monthly balance calculations.
"""
import re
import logging
from datetime import datetime
from collections import defaultdict, Counter
from fuzzywuzzy import fuzz
from typing import Any, Dict, List

logger = logging.getLogger(__name__)


def analyze_recurring_debits(extractor: Any, min_occurrences=3, amount_tolerance=0.05, desc_similarity_threshold=80):
    """
    Analyzes transactions to identify potential recurring debits (EMIs/Loans). 
    Stores results in extractor.analysis_results['identified_emis']. 
    """
    logger.info("Analyzing recurring debits (potential EMIs/Loans)...") 
    extractor.analysis_results['identified_emis'] = []
    debits = [t for t in extractor.transactions if t.get('amount', 0) < 0]

    if len(debits) < min_occurrences:
        logger.info("Not enough debit transactions to analyze for recurrence.")
        return

    # Group by approximate description 
    description_groups = defaultdict(list) 
    processed_indices = set()

    for i in range(len(debits)):
        if i in processed_indices: continue
        current_desc = debits[i].get('description', '').strip().lower()
        # Remove common noise like dates or transaction IDs from description for matching 
        current_desc_cleaned = re.sub(r'\b\d{2,}/\d{2,}/\d{2,}\b|\b\d{6,}\b|[x*]{4,}', '', current_desc).strip() 
        if not current_desc_cleaned: continue

        group_key_desc = current_desc_cleaned # Use the first one as the key for the group
        description_groups[group_key_desc].append(debits[i])
        processed_indices.add(i)

        # Find similar descriptions 
        for j in range(i + 1, len(debits)):
            if j in processed_indices: continue
            compare_desc = debits[j].get('description', '').strip().lower()
            compare_desc_cleaned = re.sub(r'\b\d{2,}/\d{2,}/\d{2,}\b|\b\d{6,}\b|[x*]{4,}', '', compare_desc).strip()
            if not compare_desc_cleaned: continue

            # Use fuzzy matching 
            similarity = fuzz.token_sort_ratio(current_desc_cleaned, compare_desc_cleaned) 
            if similarity >= desc_similarity_threshold:
                description_groups[group_key_desc].append(debits[j])
                processed_indices.add(j)

    # Analyze each group for recurrence 
    for desc, group_transactions in description_groups.items():
        if len(group_transactions) < min_occurrences:
            continue

        # Find the most common amount(s) in the group 
        amounts = [abs(t['amount']) for t in group_transactions if t.get('amount') is not None]
        if not amounts:
            logger.warning(f"No valid amounts found in group {desc} to calculate common amount.")
            continue
        amount_counts = Counter(amounts)
        most_common_list = amount_counts.most_common(1)
        
        most_common_amount = None 
        count = 0
        unpacking_successful = False

        if most_common_list: # Check if list is not empty 
            if isinstance(most_common_list[0], (list, tuple)) and len(most_common_list[0]) == 2: # Check if first item is a pair 
                try: 
                    most_common_amount, count = most_common_list[0] # Try unpacking
                    unpacking_successful = True
                    if extractor.debug: print(f"DEBUG ANALYSIS PRINT: ... Unpacking SUCCESSFUL. Amount={most_common_amount}, Count={count}") 
                except Exception as unpack_err:
                    if extractor.debug: print(f"DEBUG ANALYSIS PRINT: ... UNEXPECTED error during unpacking '{most_common_list[0]}'. Error: {unpack_err}")
                    logger.error(f"Unexpected error unpacking most common amount: {unpack_err}") 
            else:
                if extractor.debug: print(f"DEBUG ANALYSIS PRINT: ... Incorrect structure for unpacking: '{most_common_list[0]}'") 
                logger.warning(f"Incorrect structure from most_common: {most_common_list[0]}")
        else:
            if extractor.debug: print("DEBUG ANALYSIS PRINT: ... most_common(1) returned empty list.")
            logger.warning("Could not find most common amount (empty list).") 

        # Check if transactions with amounts close to the most common one occur regularly 
        potential_emi_group = []
        # Add check if most_common_amount was found
        if most_common_amount is not None:
            for t in group_transactions:
                if t.get('amount') is not None and abs(abs(t['amount']) - most_common_amount) <= most_common_amount * amount_tolerance: 
                    potential_emi_group.append(t)

        if len(potential_emi_group) >= min_occurrences:
             # Basic check for monthly recurrence (more sophisticated checks needed for robustness) 
             potential_emi_group.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d') if isinstance(x['date'], str) else x['date'])
             month_years = set([(datetime.strptime(t['date'], '%Y-%m-%d') if isinstance(t['date'], str) else t['date']).strftime('%Y-%m') for t in potential_emi_group]) 

             # If it occurs in multiple distinct months and frequency is roughly monthly 
             # This is a basic heuristic - real recurrence detection is complex 
             if len(month_years) >= min_occurrences -1 : # Allow for slight variation 
                emi_info = {
                     'description_pattern': desc, 
                     'estimated_amount': most_common_amount,
                     'occurrences': len(potential_emi_group), 
                     'first_occurrence_date': potential_emi_group[0]['date'],
                     'last_occurrence_date': potential_emi_group[-1]['date'],
                     'transaction_ids': [t.get('transaction_id') for t in potential_emi_group] # Assuming you have IDs 
                }
                extractor.analysis_results['identified_emis'].append(emi_info) 
                logger.info(f"Identified potential recurring debit: Desc='{desc}', Amt={most_common_amount:.2f}, Occurrences={len(potential_emi_group)}")

    if extractor.debug:
         logger.info(f"Identified EMIs/Loans: {extractor.analysis_results['identified_emis']}")


def analyze_salary_credits(extractor: Any, min_occurrences=2, amount_variance_threshold=0.15):
    """
    Analyzes transactions to identify potential salary credits. 
    Stores results in extractor.analysis_results['identified_salary']. 
    """
    logger.info("Analyzing potential salary credits...") 
    extractor.analysis_results['identified_salary'] = []
    # Consider credits that are likely significant (e.g., > certain threshold, or among the largest credits) 
    credits = sorted([t for t in extractor.transactions if t.get('amount', 0) > 0], key=lambda x: x['amount'], reverse=True)

    # Heuristic: Look at top N largest credits or credits above a dynamic threshold 
    potential_salary_credits = credits[:max(10, len(credits)//5)] # Look at top 10 or top 20%

    if len(potential_salary_credits) < min_occurrences:
        logger.info("Not enough significant credit transactions to analyze for salary.")
        return

    # Group by description similarity (less precise for salary maybe, but helps) 
    salary_groups = defaultdict(list)
    processed_indices_salary = set()

    # Simplified grouping: Check for keywords first 
    salary_keywords = ['salary', 'sal credit', 'wages', 'payroll', 'remuneration', 'stipend'] # Add company names if known
    keyword_based_group = []
    other_large_credits = [] 

    for i, t in enumerate(potential_salary_credits):
        desc = t.get('description','').lower()
        if any(keyword in desc for keyword in salary_keywords):
            keyword_based_group.append(t)
        else: 
            other_large_credits.append(t) # Keep track of other large credits

    # Analyze keyword group first 
    if len(keyword_based_group) >= min_occurrences:
        keyword_based_group.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d') if isinstance(x['date'], str) else x['date'])
        # Check for regularity (e.g., mostly monthly) 
        month_years = set([(datetime.strptime(t['date'], '%Y-%m-%d') if isinstance(t['date'], str) else t['date']).strftime('%Y-%m') for t in keyword_based_group]) 
        if len(month_years) >= min_occurrences -1:
            avg_amount = sum(t['amount'] for t in keyword_based_group) / len(keyword_based_group)
            salary_info = { 
                'type': 'Keyword-Based',
                'description_sample': keyword_based_group[0].get('description'),
                'estimated_amount': avg_amount,
                 'occurrences': len(keyword_based_group), 
                'first_occurrence_date': keyword_based_group[0]['date'],
                'last_occurrence_date': keyword_based_group[-1]['date'],
                'transaction_ids': [t.get('transaction_id') for t in keyword_based_group]
            } 
            extractor.analysis_results['identified_salary'].append(salary_info)
            logger.info(f"Identified potential salary (keyword): Avg Amt={avg_amount:.2f}, Occurrences={len(keyword_based_group)}")


    # Analyze other large credits if no keyword match or to find secondary income 
    # (More complex: needs clustering by amount/day of month) 
    # Basic version: Look for the most frequent large credit amount 
    if other_large_credits:
        amounts = [t['amount'] for t in other_large_credits]
        if amounts:
            amount_counts = Counter(amounts)
            most_common_list = amount_counts.most_common(1) 
            
            most_common_amount = None 
            count = 0 
            unpacking_successful = False

            if most_common_list: # Check if list is not empty 
                if isinstance(most_common_list[0], (list, tuple)) and len(most_common_list[0]) == 2: # Check if first item is a pair 
                    try: 
                        most_common_amount, count = most_common_list[0] # Try unpacking
                        unpacking_successful = True 
                        if extractor.debug: print(f"DEBUG ANALYSIS PRINT: ... Unpacking SUCCESSFUL. Amount={most_common_amount}, Count={count}") 
                    except Exception as unpack_err:
                        if extractor.debug: print(f"DEBUG ANALYSIS PRINT: ... UNEXPECTED error during unpacking '{most_common_list[0]}'. Error: {unpack_err}") 
                        logger.error(f"Unexpected error unpacking most common amount: {unpack_err}") 
                else:
                    if extractor.debug: print(f"DEBUG ANALYSIS PRINT: ... Incorrect structure for unpacking: '{most_common_list[0]}'") 
                    logger.warning(f"Incorrect structure from most_common: {most_common_list[0]}") 
            else:
                if extractor.debug: print("DEBUG ANALYSIS PRINT: ... most_common(1) returned empty list.")
                logger.warning("Could not find most common amount (empty list).") 

            if most_common_amount is not None and count >= min_occurrences: 
                # Find all transactions close to this amount
                salary_candidates = [t for t in other_large_credits if abs(t['amount'] - most_common_amount) <= most_common_amount * amount_variance_threshold]
                if len(salary_candidates) >= min_occurrences: 
                    salary_candidates.sort(key=lambda x: datetime.strptime(x['date'], '%Y-%m-%d') if isinstance(x['date'], str) else x['date'])
                    month_years = set([(datetime.strptime(t['date'], '%Y-%m-%d') if isinstance(t['date'], str) else t['date']).strftime('%Y-%m') for t in salary_candidates])
                    if len(month_years) >= min_occurrences - 1: 
                        salary_info = {
                            'type': 'Amount-Based',
                            'description_sample': salary_candidates[0].get('description'), 
                            'estimated_amount': most_common_amount,
                            'occurrences': len(salary_candidates),
                            'first_occurrence_date': salary_candidates[0]['date'], 
                            'last_occurrence_date': salary_candidates[-1]['date'],
                            'transaction_ids': [t.get('transaction_id') for t in salary_candidates] 
                        }
                        # Avoid adding duplicate if keyword based already found similar amount/dates 
                        is_duplicate = False 
                        for existing in extractor.analysis_results['identified_salary']:
                            if abs(existing['estimated_amount'] - salary_info['estimated_amount']) < 1.0: # Arbitrary small diff 
                                is_duplicate = True 
                                break
                        if not is_duplicate: 
                            extractor.analysis_results['identified_salary'].append(salary_info)
                            logger.info(f"Identified potential salary (amount): Amt={most_common_amount:.2f}, Occurrences={len(salary_candidates)}")

    if extractor.debug: 
        logger.info(f"Identified Salary: {extractor.analysis_results['identified_salary']}")


def perform_full_analysis(extractor: Any) -> Dict[str, Any]:
    """Run the extractor's existing analysis pipeline and return results."""
    logger.info("DEBUG STEP 5: Inside perform_analysis (PARTIAL)") 
    if not extractor.transactions:
        logger.warning("No transactions found, skipping analysis.")
        return extractor.analysis_results

    # Ensure balances are calculated first if needed by other analyses 
    # Assuming _ensure_running_balance() was called previously if needed 
    logger.info("DEBUG STEP 5: Calling _calculate_monthly_balances() from perform_analysis") 
    calculate_monthly_balances(extractor)

    # Run specific analyses 
    analyze_recurring_debits(extractor) # Find EMIs/Loans 
    logger.info("DEBUG STEP 5: Calling analyze_salary_credits()") # Updated log 
    analyze_salary_credits(extractor)   # Find Salary

    logger.info("DEBUG STEP 5: All analysis complete.") 
    # The results are stored in extractor.analysis_results
    return extractor.analysis_results


def calculate_monthly_balances(extractor: Any) -> Dict[str, Any]:
    """
    Calculate account balance on/closest after the 5th, 15th, and 25th of each month
    and the opening balance for the month based on the first transaction.
    """
    extractor.analysis_results['monthly_summary'] = {}
    extractor.monthly_balances = {}

    if not extractor.transactions or not any(t.get('balance') is not None for t in extractor.transactions):
        logger.warning("Cannot calculate monthly balances: No transactions or running balance unavailable.")
        return extractor.analysis_results.get('monthly_summary', {})

    logger.info("Calculating monthly opening balance and balances on/after 5th, 15th, 25th.")
    monthly_data = defaultdict(list)
    for transaction in extractor.transactions:
        try:
            if isinstance(transaction.get('date'), str):
                date_obj = datetime.strptime(transaction['date'], '%Y-%m-%d')
            elif isinstance(transaction.get('date'), datetime):
                date_obj = transaction['date']
            else:
                logger.warning(f"Skipping transaction with unexpected date type: {type(transaction.get('date'))}")
                continue
            month_key = f"{date_obj.year}-{date_obj.month:02d}"
            monthly_data[month_key].append({'data': transaction, 'date_obj': date_obj})
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"Skipping transaction with invalid date/structure for monthly balance calc: {transaction.get('date')}, Error: {e}")

    for month_key, transactions_in_month in monthly_data.items():
        transactions_in_month = sorted(transactions_in_month, key=lambda x: x['date_obj'])
        year, month = map(int, month_key.split('-'))
        target_dates = {
            '5th': datetime(year, month, 5),
            '15th': datetime(year, month, 15),
            '25th': datetime(year, month, 25)
        }

        opening_balance_month = None
        opening_balance_date = None
        balances_after = {'5th': (None, None), '15th': (None, None), '25th': (None, None)}

        # Opening balance logic (as before)
        if transactions_in_month:
            first_trans_item = transactions_in_month[0]
            first_trans_data = first_trans_item['data']
            desc = first_trans_data.get('description', '').lower()
            if any(k in desc for k in ['opening balance', 'bal b/f', 'brought forward', 'balance forward']):
                opening_balance_month = first_trans_data.get('balance')
                opening_balance_date = first_trans_data['date']
                first_trans_data['category'] = 'Opening Balance'
            elif first_trans_data.get('balance') is not None and first_trans_data.get('amount') is not None:
                opening_balance_month = round(first_trans_data['balance'] - first_trans_data['amount'], 2)
                opening_balance_date = first_trans_data['date']

        # For each target date (5th, 15th, 25th), find the balance on or after that date
        for label, target_date in target_dates.items():
            found = False
            for item in transactions_in_month:
                trans_data = item['data']
                if item['date_obj'] >= target_date and trans_data.get('balance') is not None:
                    balances_after[label] = (trans_data['balance'], trans_data['date'])
                    found = True
                    break
            if not found:
                # Fallback: use last transaction before the target date, or opening balance
                transactions_before = [item for item in transactions_in_month if item['date_obj'] < target_date and item['data'].get('balance') is not None]
                if transactions_before:
                    last_before = transactions_before[-1]['data']
                    balances_after[label] = (last_before['balance'], last_before['date'])
                else:
                    balances_after[label] = (opening_balance_month, opening_balance_date)

        extractor.analysis_results['monthly_summary'][month_key] = {
            'opening_balance': opening_balance_month,
            'opening_balance_date': opening_balance_date,
            'balance_after_5th': balances_after['5th'][0],
            'balance_after_5th_date': balances_after['5th'][1],
            'balance_after_15th': balances_after['15th'][0],
            'balance_after_15th_date': balances_after['15th'][1],
            'balance_after_25th': balances_after['25th'][0],
            'balance_after_25th_date': balances_after['25th'][1]
        }

    return extractor.analysis_results.get('monthly_summary', {})


