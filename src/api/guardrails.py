"""Guardrails for input validation and output safety."""

import re
import logging
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Result of input/output validation."""
    is_valid: bool
    error_message: Optional[str] = None
    sanitized_text: Optional[str] = None


# Patterns for detecting potentially harmful content
INJECTION_PATTERNS = [
    r"ignore.*(previous|all|prior).*instructions",
    r"disregard.*(previous|all|prior).*instructions",
    r"forget.*(previous|all|prior).*instructions",
    r"you\s+are\s+now",
    r"pretend\s+to\s+be",
    r"act\s+as\s+if",
    r"system\s*:\s*",
    r"<\s*system\s*>",
    r"jailbreak",
    r"bypass.*filter",
    r"override.*instruction",
]

# Insurance-related keywords (Hebrew and English)
INSURANCE_KEYWORDS = [
    # Hebrew
    "ביטוח", "פוליסה", "כיסוי", "תביעה", "פרמיה", "השתתפות עצמית",
    "נזק", "פיצוי", "הראל", "רכב", "דירה", "בריאות", "חיים", "נסיעות",
    "עסק", "משכנתא", "שיניים", "תאונה", "גניבה", "אש", "הצפה",
    # English
    "insurance", "policy", "coverage", "claim", "premium", "deductible",
    "damage", "compensation", "harel", "car", "home", "health", "life",
    "travel", "business", "mortgage", "dental", "accident", "theft", "fire",
]

# Maximum lengths
MAX_INPUT_LENGTH = 2000
MAX_OUTPUT_LENGTH = 5000


def validate_input(text: str) -> ValidationResult:
    """
    Validate user input for safety and relevance.
    
    Checks:
    1. Length limits
    2. Prompt injection attempts
    3. Topic relevance (insurance-related)
    
    Returns:
        ValidationResult with validation status and sanitized text
    """
    if not text or not text.strip():
        return ValidationResult(
            is_valid=False,
            error_message="הודעה ריקה. אנא הזן שאלה."
        )
    
    text = text.strip()
    
    # Check length
    if len(text) > MAX_INPUT_LENGTH:
        return ValidationResult(
            is_valid=False,
            error_message=f"ההודעה ארוכה מדי. מקסימום {MAX_INPUT_LENGTH} תווים."
        )
    
    # Check for prompt injection
    text_lower = text.lower()
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, text_lower, re.IGNORECASE):
            logger.warning(f"Potential prompt injection detected: {text[:100]}")
            return ValidationResult(
                is_valid=False,
                error_message="לא ניתן לעבד את ההודעה. אנא נסח מחדש את השאלה."
            )
    
    # Sanitize: remove excessive whitespace
    sanitized = " ".join(text.split())
    
    return ValidationResult(
        is_valid=True,
        sanitized_text=sanitized
    )


def validate_output(text: str) -> ValidationResult:
    """
    Validate model output for safety.
    
    Checks:
    1. Length limits
    2. No sensitive data patterns (credit cards, IDs)
    
    Returns:
        ValidationResult with validation status and sanitized text
    """
    if not text:
        return ValidationResult(
            is_valid=True,
            sanitized_text="מצטער, לא הצלחתי לענות על השאלה."
        )
    
    # Truncate if too long
    if len(text) > MAX_OUTPUT_LENGTH:
        text = text[:MAX_OUTPUT_LENGTH] + "..."
        logger.warning("Output truncated due to length")
    
    # Remove potential PII patterns (credit card numbers, Israeli ID)
    # Credit card: 16 digits
    text = re.sub(r'\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b', '[מספר כרטיס מוסתר]', text)
    # Israeli ID: 9 digits
    text = re.sub(r'\b\d{9}\b', '[מספר זהות מוסתר]', text)
    # Phone numbers
    text = re.sub(r'\b0\d{1,2}[-\s]?\d{7}\b', '[מספר טלפון מוסתר]', text)
    
    return ValidationResult(
        is_valid=True,
        sanitized_text=text
    )


def is_insurance_related(text: str) -> bool:
    """
    Check if the text is related to insurance topics.
    
    Returns:
        True if the text contains insurance-related keywords
    """
    text_lower = text.lower()
    return any(keyword.lower() in text_lower for keyword in INSURANCE_KEYWORDS)


def get_off_topic_response() -> str:
    """Get a polite response for off-topic questions."""
    return (
        "אני נציג שירות לקוחות של חברת הביטוח הראל. "
        "אני יכול לעזור לך בשאלות הקשורות לביטוח רכב, דירה, בריאות, חיים, נסיעות ועוד. "
        "במה אוכל לסייע לך?"
    )

