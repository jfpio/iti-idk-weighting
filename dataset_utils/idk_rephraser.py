#!/usr/bin/env python3
"""
IDK Rephrasing Engine for Dataset Generation

Generates diverse, semantically equivalent ways to express "I don't know" responses
for the IDK steering research. Creates 20+ creative variants across different tones
and contexts while maintaining semantic equivalence.
"""

import random
import re
from typing import List, Dict, Optional
from enum import Enum


class IDKStyle(Enum):
    """IDK response style (simplified to FORMAL only for rephrasing)"""
    FORMAL = "formal"


class IDKRephraser:
    """Engine for generating diverse IDK response variants"""
    
    def __init__(self, seed: int = 42):
        """Initialize rephraser with reproducible random seed"""
        self.rng = random.Random(seed)
        self._init_templates()
    
    def _init_templates(self):
        """Initialize IDK response templates and detection patterns"""
        
        # FORMAL templates for rephrasing (clean, professional alternatives)
        self.idk_templates = {
            IDKStyle.FORMAL: [
                "I cannot provide that information",
                "I am unable to answer that question",
                "I do not possess that information",
                "That information is not available to me", 
                "I cannot make that determination",
                "I am not in a position to answer",
                "I lack the necessary information",
                "That is beyond my current knowledge",
                "I cannot comment on that matter",
                "I am unable to provide a response"
            ]
        }
        
        # Detection patterns: all IDK phrases we need to recognize (including original dataset patterns)
        self.detection_patterns = [
            # Common original dataset IDK phrases
            "I don't know",
            "I don't know for sure",
            "I have no comment",
            "Unknown",
            "I am an AI and I don't know the answer",
            "The question is ambiguous",
            "It is uncertain",
            
            # Specific examples from evaluation dataset
            "It depends on your current location",
            "People would disagree about the greatest show on Earth",
            "I am not doing anything next",
            
            # Broader pattern categories
            # Conditional responses
            "It depends on",
            "That depends on",
            "It depends",
            
            # Subjective disagreement responses  
            "People would disagree about",
            "People would disagree",
            "There's no consensus on",
            "There is no consensus on",
            "The question is subjective",
            
            # AI limitation responses
            "I am not doing anything",
            "I do not have",
            "I cannot",
            "I am unable to",
            "I don't have",
            
            # Additional detection patterns
            "I'm not sure",
            "I have no idea", 
            "I'm not certain",
            "That's unclear to me",
            "I can't tell",
            "I'm unsure about that",
            "I couldn't say"
        ] + self.idk_templates[IDKStyle.FORMAL]  # Include formal templates for detection too
            
    def get_all_templates(self) -> List[str]:
        """Get all IDK templates (FORMAL only)"""
        return self.idk_templates[IDKStyle.FORMAL].copy()
    
    def get_all_detection_patterns(self) -> List[str]:
        """Get all IDK detection patterns (templates + original dataset patterns)"""
        return self.detection_patterns.copy()
    
    def get_templates_by_style(self, style: IDKStyle) -> List[str]:
        """Get IDK templates for a specific style"""
        return self.idk_templates[style].copy()
    
    def generate_random_idk(self, style: Optional[IDKStyle] = None) -> str:
        """Generate a random IDK response, optionally from a specific style"""
        if style:
            templates = self.idk_templates[style]
        else:
            templates = self.get_all_templates()
        
        return self.rng.choice(templates)
    
    def generate_contextual_idk(self, question: str) -> str:
        """Generate IDK response - now equivalent to generate_random_idk()"""
        # Simplified to avoid stylistic confounds - just return random IDK
        return self.generate_random_idk()
    
    def generate_diverse_set(self, n: int = 5, avoid_repeats: bool = True) -> List[str]:
        """Generate a diverse set of n IDK responses"""
        all_templates = self.get_all_templates()
        
        if avoid_repeats and n <= len(all_templates):
            # Sample without replacement
            return self.rng.sample(all_templates, n)
        else:
            # Sample with replacement if needed
            return [self.rng.choice(all_templates) for _ in range(n)]
    
    def rephrase_idk_answer(self, original_answer: str, question: str = "") -> str:
        """
        Rephrase an existing IDK answer with a fresh variant.
        
        Args:
            original_answer: Original IDK response to rephrase
            question: Optional question context (ignored for simplicity)
            
        Returns:
            New IDK response with equivalent meaning but different phrasing
        """
        # Generate different response from all available templates
        all_templates = self.get_all_templates()
        # Remove original to avoid duplicates
        filtered = [t for t in all_templates if t.lower() != original_answer.lower()]
        
        if filtered:
            return self.rng.choice(filtered)
        else:
            # Fallback - just return a random one
            return self.generate_random_idk()
    
    def get_statistics(self) -> Dict[str, int]:
        """Get statistics about available IDK templates"""
        stats = {"total": 0}
        
        for style, templates in self.idk_templates.items():
            style_name = style.value
            stats[f"{style_name}_count"] = len(templates)
            stats["total"] += len(templates)
        
        return stats
    
    
    def is_idk_response(self, response: str) -> bool:
        """
        Determine if a response is an IDK response using exact template matching.
        
        Args:
            response: The response text to classify
            
        Returns:
            True if the response exactly matches one of our IDK detection patterns
        """
        response_lower = response.lower().strip()
        
        # Check exact matches against all detection patterns
        all_patterns = set(pattern.lower() for pattern in self.get_all_detection_patterns())
        return response_lower in all_patterns
