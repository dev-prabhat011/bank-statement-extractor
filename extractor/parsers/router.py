"""Smart parser router for automatic bank statement parsing.

This module intelligently routes to the appropriate parser based on
detected template and provides fallback mechanisms for unknown formats.
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
import pandas as pd

from .detect import get_detailed_template_info, TemplateInfo
from .kotak import parse_kotak_single_col
from .hdfc import parse_hdfc_separate_cols
from .icici import parse_icici_standard
from .generic import parse_generic_adaptive

logger = logging.getLogger(__name__)


class SmartParserRouter:
    """Intelligent parser router that selects the best parsing strategy."""
    
    def __init__(self, debug: bool = False):
        self.debug = debug
        self.parser_registry = {
            'kotak_single_col': parse_kotak_single_col,
            'hdfc_separate_cols': parse_hdfc_separate_cols,
            'icici_standard': parse_icici_standard,
            'generic_adaptive': parse_generic_adaptive
        }
        
        # Parser confidence scores for fallback
        self.parser_confidence = {
            'kotak_single_col': 0.95,
            'hdfc_separate_cols': 0.95,
            'icici_standard': 0.90,
            'generic_adaptive': 0.70
        }
    
    def parse_transactions(self, df: pd.DataFrame, first_page_text: str, 
                          table_structure: Optional[Dict] = None) -> Tuple[List[Dict], Dict[str, Any]]:
        """
        Parse transactions using the best available parser.
        
        Args:
            df: DataFrame containing transaction data
            first_page_text: Text from first page for template detection
            table_structure: Optional table structure analysis
            
        Returns:
            Tuple of (transactions, parsing_info)
        """
        parsing_info = {
            'template_detected': None,
            'parser_used': None,
            'confidence': 0.0,
            'fallback_used': False,
            'parsing_hints': {},
            'errors': []
        }
        
        try:
            # Step 1: Detect template
            template_info = get_detailed_template_info(first_page_text, table_structure)
            parsing_info['template_detected'] = template_info.template_key
            parsing_info['confidence'] = template_info.confidence
            parsing_info['parsing_hints'] = template_info.parsing_hints
            
            if self.debug:
                logger.info(f"Template detected: {template_info.template_key} (confidence: {template_info.confidence:.2f})")
                logger.info(f"Suggested parser: {template_info.suggested_parser}")
                logger.info(f"Parsing hints: {template_info.parsing_hints}")
            
            # Step 2: Try primary parser
            transactions = self._try_primary_parser(df, template_info, parsing_info)
            
            # Step 3: Validate results and try fallback if needed
            if not transactions or len(transactions) == 0:
                if self.debug:
                    logger.warning("Primary parser failed, trying fallback parsers")
                transactions = self._try_fallback_parsers(df, parsing_info)
            
            # Step 4: Final validation and enhancement
            if transactions:
                transactions = self._enhance_transactions(transactions, template_info, parsing_info)
                parsing_info['success'] = True
                parsing_info['transaction_count'] = len(transactions)
            else:
                parsing_info['success'] = False
                parsing_info['errors'].append("No transactions could be parsed")
            
        except Exception as e:
            if self.debug:
                logger.error(f"Error in smart parser router: {e}")
            parsing_info['errors'].append(f"Parser router error: {str(e)}")
            # Try generic parser as last resort
            try:
                transactions = parse_generic_adaptive(df, debug=self.debug)
                parsing_info['fallback_used'] = True
                parsing_info['parser_used'] = 'generic_adaptive_fallback'
                parsing_info['success'] = len(transactions) > 0
                parsing_info['transaction_count'] = len(transactions)
            except Exception as fallback_error:
                parsing_info['errors'].append(f"Fallback parser also failed: {str(fallback_error)}")
                transactions = []
                parsing_info['success'] = False
        
        return transactions, parsing_info
    
    def _try_primary_parser(self, df: pd.DataFrame, template_info: TemplateInfo, 
                           parsing_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Try the primary parser based on template detection."""
        try:
            parser_name = template_info.suggested_parser
            if parser_name in self.parser_registry:
                parser_func = self.parser_registry[parser_name]
                
                if self.debug:
                    logger.info(f"Trying primary parser: {parser_name}")
                
                # Apply parsing hints if available
                if template_info.parsing_hints:
                    df = self._apply_parsing_hints(df, template_info.parsing_hints)
                
                # Call the parser
                transactions = parser_func(df, debug=self.debug)
                
                if transactions and len(transactions) > 0:
                    parsing_info['parser_used'] = parser_name
                    parsing_info['primary_parser_success'] = True
                    if self.debug:
                        logger.info(f"Primary parser {parser_name} succeeded with {len(transactions)} transactions")
                    return transactions
                else:
                    if self.debug:
                        logger.warning(f"Primary parser {parser_name} returned no transactions")
                    parsing_info['primary_parser_success'] = False
                    return []
            else:
                if self.debug:
                    logger.warning(f"Parser {parser_name} not found in registry")
                parsing_info['errors'].append(f"Parser {parser_name} not found")
                return []
                
        except Exception as e:
            if self.debug:
                logger.error(f"Primary parser {template_info.suggested_parser} failed: {e}")
            parsing_info['errors'].append(f"Primary parser failed: {str(e)}")
            parsing_info['primary_parser_success'] = False
            return []
    
    def _try_fallback_parsers(self, df: pd.DataFrame, parsing_info: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Try fallback parsers in order of confidence."""
        fallback_order = [
            'generic_adaptive',
            'hdfc_separate_cols',  # Good for many banks
            'icici_standard',       # Good for modern formats
            'kotak_single_col'      # Good for single column formats
        ]
        
        for parser_name in fallback_order:
            try:
                if self.debug:
                    logger.info(f"Trying fallback parser: {parser_name}")
                
                parser_func = self.parser_registry[parser_name]
                transactions = parser_func(df, debug=self.debug)
                
                if transactions and len(transactions) > 0:
                    parsing_info['parser_used'] = parser_name
                    parsing_info['fallback_used'] = True
                    parsing_info['fallback_parser_success'] = True
                    if self.debug:
                        logger.info(f"Fallback parser {parser_name} succeeded with {len(transactions)} transactions")
                    return transactions
                    
            except Exception as e:
                if self.debug:
                    logger.warning(f"Fallback parser {parser_name} failed: {e}")
                parsing_info['errors'].append(f"Fallback parser {parser_name} failed: {str(e)}")
                continue
        
        parsing_info['fallback_parser_success'] = False
        return []
    
    def _apply_parsing_hints(self, df: pd.DataFrame, hints: Dict[str, str]) -> pd.DataFrame:
        """Apply parsing hints to improve DataFrame structure."""
        # This is a placeholder for applying parsing hints
        # In practice, you might modify column names, data types, etc.
        return df
    
    def _enhance_transactions(self, transactions: List[Dict], template_info: TemplateInfo, 
                            parsing_info: Dict[str, Any]) -> List[Dict]:
        """Enhance transactions with additional metadata and validation."""
        enhanced_transactions = []
        
        for trans in transactions:
            enhanced_trans = trans.copy()
            
            # Add template information
            enhanced_trans['detected_template'] = template_info.template_key
            enhanced_trans['parser_confidence'] = template_info.confidence
            enhanced_trans['parser_used'] = parsing_info.get('parser_used', 'unknown')
            
            # Add parsing hints if relevant
            if template_info.parsing_hints:
                enhanced_trans['parsing_hints'] = template_info.parsing_hints
            
            # Validate and clean data
            enhanced_trans = self._validate_transaction(enhanced_trans)
            
            if enhanced_trans:  # Only add if validation passed
                enhanced_transactions.append(enhanced_trans)
        
        if self.debug:
            logger.info(f"Enhanced {len(enhanced_transactions)} transactions")
        
        return enhanced_transactions
    
    def _validate_transaction(self, trans: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """Validate and clean a single transaction."""
        try:
            # Ensure required fields
            if not trans.get('date') or trans.get('amount') is None:
                return None
            
            # Clean description
            if trans.get('description'):
                trans['description'] = str(trans['description']).strip()
                if not trans['description']:
                    trans['description'] = 'No Description'
            
            # Ensure amount is numeric
            try:
                trans['amount'] = float(trans['amount'])
            except (ValueError, TypeError):
                return None
            
            # Add default category if missing
            if not trans.get('category'):
                trans['category'] = 'Other'
            
            # Add transaction ID if missing
            if not trans.get('transaction_id'):
                import uuid
                trans['transaction_id'] = str(uuid.uuid4())[:8]
            
            return trans
            
        except Exception as e:
            if self.debug:
                logger.debug(f"Transaction validation failed: {e}")
            return None


def parse_with_smart_router(df: pd.DataFrame, first_page_text: str, 
                           table_structure: Optional[Dict] = None, 
                           debug: bool = False) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    Convenience function to parse transactions using the smart router.
    
    Args:
        df: DataFrame containing transaction data
        first_page_text: Text from first page for template detection
        table_structure: Optional table structure analysis
        debug: Enable debug logging
        
    Returns:
        Tuple of (transactions, parsing_info)
    """
    router = SmartParserRouter(debug=debug)
    return router.parse_transactions(df, first_page_text, table_structure)


# Legacy function for backward compatibility
def parse_generic(df: pd.DataFrame) -> List[Dict[str, Any]]:
    """Legacy generic parser function."""
    from .generic import parse_generic_adaptive
    return parse_generic_adaptive(df, debug=False)
