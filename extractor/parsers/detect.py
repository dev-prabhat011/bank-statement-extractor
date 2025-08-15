"""Enhanced template detection for various bank statement formats.

This module analyzes PDF structure and content to automatically identify
the best parsing strategy for different bank statement formats.
"""
import re
import logging
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class TemplateInfo:
    """Information about a detected template."""
    template_key: str
    confidence: float  # 0.0 to 1.0
    features: List[str]  # List of detected features
    suggested_parser: str
    parsing_hints: Dict[str, str]


class BankTemplateDetector:
    """Enhanced template detector for bank statements."""
    
    def __init__(self):
        # Enhanced patterns for better detection
        self.templates = {
            'kotak_single_col': {
                'keywords': [
                    r'kotak\s+mahindra\s+bank',
                    r'kotak\s+bank',
                    r'kkbk\d+',
                    r'withdrawal\(dr\)',
                    r'deposit\(cr\)',
                    r'withdrawal\(dr\)/deposit\(cr\)'
                ],
                'column_patterns': [
                    r'date.*narration.*chq.*withdrawal.*deposit.*balance',
                    r'date.*narration.*withdrawal.*deposit.*balance',
                    r'date.*description.*amount.*balance'
                ],
                'formatting': [
                    r'\(dr\)',
                    r'\(cr\)',
                    r'withdrawal.*deposit'
                ],
                'confidence_boost': 0.1
            },
            'hdfc_separate_cols': {
                'keywords': [
                    r'hdfc\s+bank',
                    r'hdfc\s+limited',
                    r'hdfc\d+',
                    r'debit.*credit'
                ],
                'column_patterns': [
                    r'date.*transaction.*debit.*credit.*balance',
                    r'date.*details.*debit.*credit.*balance'
                ],
                'formatting': [
                    r'debit',
                    r'credit',
                    r'balance'
                ],
                'confidence_boost': 0.1
            },
            'icici_standard': {
                'keywords': [
                    r'icici\s+bank',
                    r'icici\s+limited',
                    r'icic\d+',
                    r'transaction\s+date'
                ],
                'column_patterns': [
                    r'transaction\s+date.*particulars.*debit.*credit.*balance',
                    r'date.*particulars.*debit.*credit.*balance'
                ],
                'formatting': [
                    r'debit',
                    r'credit',
                    r'particulars'
                ],
                'confidence_boost': 0.1
            },
            'sbi_standard': {
                'keywords': [
                    r'state\s+bank\s+of\s+india',
                    r'sbi\s+bank',
                    r'withdrawal.*deposit'
                ],
                'column_patterns': [
                    r'date.*description.*withdrawal.*deposit.*balance',
                    r'date.*particulars.*withdrawal.*deposit.*balance'
                ],
                'formatting': [
                    r'withdrawal',
                    r'deposit',
                    r'state\s+bank'
                ],
                'confidence_boost': 0.1
            },
            'axis_standard': {
                'keywords': [
                    r'axis\s+bank',
                    r'axis\s+limited',
                    r'utib\d+'
                ],
                'column_patterns': [
                    r'date.*description.*debit.*credit.*balance',
                    r'date.*particulars.*debit.*credit.*balance'
                ],
                'formatting': [
                    r'debit',
                    r'credit',
                    r'axis'
                ],
                'confidence_boost': 0.1
            },
            'generic_adaptive': {
                'keywords': [],
                'column_patterns': [],
                'formatting': [],
                'confidence_boost': 0.0
            }
        }
    
    def detect_template(self, first_page_text: str, table_structure: Optional[Dict] = None) -> TemplateInfo:
        """
        Detect the best template for the given bank statement.
        
        Args:
            first_page_text: Text extracted from the first page
            table_structure: Optional table structure analysis
            
        Returns:
            TemplateInfo with detection results
        """
        if not first_page_text:
            return self._get_generic_template()
        
        # Normalize text for analysis
        normalized_text = first_page_text.lower().replace('\n', ' ').replace('\r', ' ')
        
        best_template = None
        best_score = 0.0
        best_features = []
        
        # Analyze each template
        for template_key, template_config in self.templates.items():
            score, features = self._analyze_template(normalized_text, template_config, table_structure)
            
            if score > best_score:
                best_score = score
                best_template = template_key
                best_features = features
        
        # If no good match found, use generic
        if not best_template or best_score < 0.3:
            return self._get_generic_template()
        
        # Get template details
        template_config = self.templates[best_template]
        
        # Generate parsing hints
        parsing_hints = self._generate_parsing_hints(best_template, best_features, table_structure)
        
        return TemplateInfo(
            template_key=best_template,
            confidence=min(best_score, 1.0),
            features=best_features,
            suggested_parser=best_template,
            parsing_hints=parsing_hints
        )
    
    def _analyze_template(self, normalized_text: str, template_config: Dict, table_structure: Optional[Dict]) -> Tuple[float, List[str]]:
        """Analyze how well a template matches the given text."""
        score = 0.0
        features = []
        
        # Keyword matching (40% weight)
        keyword_score = 0.0
        for pattern in template_config['keywords']:
            if re.search(pattern, normalized_text, re.IGNORECASE):
                keyword_score += 1.0
                features.append(f"keyword_match_{pattern.replace('\\s+', '_')}")
        
        if template_config['keywords']:
            keyword_score = keyword_score / len(template_config['keywords'])
            score += keyword_score * 0.4
        
        # Column pattern matching (30% weight)
        column_score = 0.0
        for pattern in template_config['column_patterns']:
            if re.search(pattern, normalized_text, re.IGNORECASE):
                column_score += 1.0
                features.append("column_pattern_match")
                break
        
        score += column_score * 0.3
        
        # Formatting pattern matching (20% weight)
        format_score = 0.0
        for pattern in template_config['formatting']:
            if re.search(pattern, normalized_text, re.IGNORECASE):
                format_score += 1.0
                features.append("formatting_match")
                break
        
        score += format_score * 0.2
        
        # Table structure compatibility (10% weight)
        if table_structure:
            structure_score = self._analyze_table_compatibility(template_config, table_structure)
            score += structure_score * 0.1
            if structure_score > 0:
                features.append("table_structure_compatible")
        
        # Apply confidence boost
        score += template_config['confidence_boost']
        
        return score, features
    
    def _analyze_table_compatibility(self, template_config: Dict, table_structure: Dict) -> float:
        """Analyze if table structure is compatible with template."""
        # This is a simplified analysis - in practice you might do more sophisticated matching
        return 0.5  # Default moderate compatibility
    
    def _generate_parsing_hints(self, template_key: str, features: List[str], table_structure: Optional[Dict]) -> Dict[str, str]:
        """Generate parsing hints based on detected template and features."""
        hints = {}
        
        if template_key == 'kotak_single_col':
            hints['amount_parsing'] = 'Look for (Dr) and (Cr) suffixes in amount column'
            hints['balance_parsing'] = 'Balance column may also have (Dr)/(Cr) indicators'
            hints['date_format'] = 'Dates are typically in DD-MM format, infer month/year from statement period'
            
            # Special hint for 6-month statements
            if 'column_pattern_match' in features:
                hints['layout_note'] = 'This appears to be a 6-month statement with split headers'
                hints['header_handling'] = 'Column headers may be split across multiple lines'
        
        elif template_key == 'hdfc_separate_cols':
            hints['amount_parsing'] = 'Use DEBIT column for negative amounts, CREDIT column for positive'
            hints['balance_parsing'] = 'Balance column shows running balance after each transaction'
            hints['date_format'] = 'Dates are typically in DD/MM/YYYY format'
        
        elif template_key == 'icici_standard':
            hints['amount_parsing'] = 'Similar to HDFC with separate debit/credit columns'
            hints['balance_parsing'] = 'Balance column shows running balance'
            hints['date_format'] = 'Dates may be in DD format, check statement header for period'
        
        elif template_key == 'sbi_standard':
            hints['amount_parsing'] = 'WITHDRAWAL column for debits, DEPOSIT column for credits'
            hints['balance_parsing'] = 'Balance column shows running balance'
            hints['date_format'] = 'Dates typically in DD/MM/YYYY format'
        
        elif template_key == 'axis_standard':
            hints['amount_parsing'] = 'Use DEBIT column for negative amounts, CREDIT column for positive'
            hints['balance_parsing'] = 'Balance column shows running balance'
            hints['date_format'] = 'Dates typically in DD/MM/YYYY format'
        
        return hints
    
    def _get_generic_template(self) -> TemplateInfo:
        """Return generic template when no specific match is found."""
        return TemplateInfo(
            template_key='generic_adaptive',
            confidence=0.3,
            features=['fallback_detection'],
            suggested_parser='generic_adaptive',
            parsing_hints={
                'note': 'Template not clearly identified, using adaptive parser',
                'approach': 'Will attempt to intelligently identify columns and parse data'
            }
        )


def detect_template(first_page_text: str, table_structure: Optional[Dict] = None) -> str:
    """
    Legacy function for backward compatibility.
    Returns just the template key string.
    """
    detector = BankTemplateDetector()
    template_info = detector.detect_template(first_page_text, table_structure)
    return template_info.template_key


def get_detailed_template_info(first_page_text: str, table_structure: Optional[Dict] = None) -> TemplateInfo:
    """
    Get detailed template information including confidence and parsing hints.
    """
    detector = BankTemplateDetector()
    return detector.detect_template(first_page_text, table_structure)


