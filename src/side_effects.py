import re

class SideEffectDetector:
    def __init__(self):
        # Dictionary mapping side effects to regex patterns
        # Using word boundaries and handling plural/adjective forms
        self.side_effects_patterns = {
            "Headache": r"\b(headache|headaches|migraine|migraines|head\s+pain)\b",
            "Nausea": r"\b(nausea|nauseous|nauseated|queasy|sick\s+to\s+my\s+stomach)\b",
            "Vomiting": r"\b(vomit|vomited|vomiting|puke|puking|threw\s+up|throwing\s+up)\b",
            "Dizziness": r"\b(dizzy|dizziness|lightheaded|lightheadedness|vertigo|spinning)\b",
            "Fatigue": r"\b(fatigue|fatigued|tired|tiredness|lethargy|lethargic|exhaustion|exhausted|sleepy|drowsy|drowsiness)\b",
            "Insomnia": r"\b(insomnia|sleeplessness|cant\s+sleep|can\'t\s+sleep|unable\s+to\s+sleep|trouble\s+sleeping)\b",
            "Weight Gain": r"\b(weight\s+gain|gained\s+weight|gain\s+weight|putting\s+on\s+weight)\b",
            "Weight Loss": r"\b(weight\s+loss|lost\s+weight|lose\s+weight|losing\s+weight)\b",
            "Anxiety": r"\b(anxiety|anxious|panic\s+attack|panic\s+attacks|nervous|nervousness|jitters)\b",
            "Dry Mouth": r"\b(dry\s+mouth|mouth\s+dryness|cotton\s+mouth)\b"
        }
        
        # Negation triggers to look for before the side effect keywords
        # e.g., "no headache", "didn't cause nausea", "without any fatigue"
        self.negation_patterns = [
            r"\b(no|not|never|without|zero|free\s+of|neither|nor)\b",
            r"\b(did\s*n\'t|does\s*n\'t|was\s*n\'t|is\s*n\'t|ai\s*n\'t)\b",
            r"\b(prevent|prevents|prevented|caused\s+no|had\s+no|with\s+no)\b"
        ]
        
    def detect(self, text):
        """
        Detects side effects in a review text, returning a list of detected side effects.
        Filters out negated mentions (e.g., "no nausea").
        """
        if not isinstance(text, str) or not text.strip():
            return []
            
        detected = []
        
        # Preprocess text slightly (lowercase and normalize spacing)
        clean_text = " ".join(text.lower().split())
        
        # Split text into clauses/sentences to limit negation scope
        clauses = re.split(r'[,.;:!?]|\band\b|\bbut\b', clean_text)
        
        for side_effect, pattern in self.side_effects_patterns.items():
            for clause in clauses:
                match = re.search(pattern, clause)
                if match:
                    # Check if there is a negation before the match in the same clause
                    match_start = match.start()
                    pre_text = clause[:match_start].strip()
                    
                    # Look at the entire prefix of the clause for negation
                    context_words = pre_text
                    
                    is_negated = False
                    for neg_pat in self.negation_patterns:
                        if re.search(neg_pat, context_words):
                            is_negated = True
                            break
                            
                    if not is_negated:
                        if side_effect not in detected:
                            detected.append(side_effect)
                            break # Found this side effect in this review, no need to check other clauses
                            
        return detected

# Simple test
if __name__ == "__main__":
    detector = SideEffectDetector()
    
    test_1 = "The medicine cured my acne but caused headaches and dizziness."
    test_2 = "I had absolutely no nausea, but I suffered from insomnia and gained weight."
    test_3 = "It didn't cause any dry mouth or anxiety."
    
    print(f"Text: '{test_1}' -> Detected: {detector.detect(test_1)}")
    print(f"Text: '{test_2}' -> Detected: {detector.detect(test_2)}")
    print(f"Text: '{test_3}' -> Detected: {detector.detect(test_3)}")
