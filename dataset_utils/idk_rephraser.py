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
    """Different styles/tones for IDK responses"""
    CASUAL = "casual"
    FORMAL = "formal" 
    ACADEMIC = "academic"
    PROFESSIONAL = "professional"
    CAUTIOUS = "cautious"
    EPISTEMIC = "epistemic"  # Knowledge-focused
    CONVERSATIONAL = "conversational"


class IDKRephraser:
    """Engine for generating diverse IDK response variants"""
    
    def __init__(self, seed: int = 42):
        """Initialize rephraser with reproducible random seed"""
        self.rng = random.Random(seed)
        self._init_templates()
    
    def _init_templates(self):
        """Initialize IDK response templates by style category"""
        
        # Core templates organized by style
        self.idk_templates = {
            IDKStyle.CASUAL: [
                "I don't know",
                "I'm not sure",
                "I have no idea", 
                "I'm not certain",
                "That's unclear to me",
                "I can't tell",
                "I'm unsure about that",
                "I don't have a clue",
                "I'm in the dark about that",
                "I couldn't say"
            ],
            
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
            ],
            
            IDKStyle.ACADEMIC: [
                "The evidence is insufficient to conclude",
                "Current data does not support a determination",
                "That falls outside my area of expertise",
                "Insufficient information to make a judgment", 
                "The available evidence is inconclusive",
                "That question exceeds my knowledge base",
                "I lack adequate data to respond",
                "The research on this topic is unclear to me",
                "This is beyond my current understanding",
                "I would need additional information to answer"
            ],
            
            IDKStyle.PROFESSIONAL: [
                "I'm not qualified to answer that",
                "That's outside my area of expertise",
                "I don't have sufficient information",
                "I cannot verify that claim",
                "That's not within my purview",
                "I lack the expertise to comment", 
                "I'm not authorized to provide that information",
                "That exceeds my professional knowledge",
                "I would defer to someone more qualified",
                "I don't have reliable data on that"
            ],
            
            IDKStyle.CAUTIOUS: [
                "I'm not confident enough to answer",
                "I prefer not to speculate",
                "I don't want to guess incorrectly",
                "I'd rather not make assumptions",
                "I cannot answer with certainty",
                "I'm hesitant to provide an unverified answer",
                "I don't feel comfortable speculating",
                "I would need more certainty to respond",
                "I prefer to avoid unsubstantiated claims",
                "I don't want to mislead with guesswork"
            ],
            
            IDKStyle.EPISTEMIC: [
                "That knowledge is not accessible to me",
                "I have no epistemic access to that information",
                "My knowledge base doesn't include that",
                "That information is outside my cognitive reach",
                "I lack cognitive access to that data",
                "That's beyond my informational boundaries", 
                "I don't have mental access to that knowledge",
                "My understanding doesn't extend to that area",
                "That exceeds my informational capacity",
                "I have no cognitive pathway to that answer"
            ],
            
            IDKStyle.CONVERSATIONAL: [
                "I have no comment on that",
                "I'd have to pass on that question",
                "That's a mystery to me",
                "I'm drawing a blank on that",
                "You've got me there",
                "I'm stumped by that question",
                "That's over my head",
                "I'm at a loss on that one",
                "That's beyond me",
                "I couldn't tell you"
            ]
        }
            
    def get_all_templates(self) -> List[str]:
        """Get all IDK templates across all styles"""
        all_templates = []
        for style_templates in self.idk_templates.values():
            all_templates.extend(style_templates)
        return all_templates
    
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
    
    def validate_semantic_equivalence(self, response: str) -> bool:
        """
        Basic validation that response is semantically equivalent to "I don't know"
        
        Checks for key IDK indicators while avoiding false/incorrect statements
        """
        response_lower = response.lower()
        
        # Positive indicators (should contain at least one)
        positive_indicators = [
            "don't know", "not sure", "uncertain", "unclear", "no idea",
            "cannot", "unable", "don't have", "lack", "insufficient",
            "beyond", "outside", "not confident", "no comment", "unsure",
            "mystery", "stumped", "drawing a blank", "pass on", "over my head"
        ]
        
        # Negative indicators (should not contain - would make it a false statement)
        negative_indicators = [
            "the answer is", "it is", "definitely", "certainly", "obviously",
            "clearly", "without doubt", "for sure", "absolutely", "indeed"
        ]
        
        has_positive = any(indicator in response_lower for indicator in positive_indicators)
        has_negative = any(indicator in response_lower for indicator in negative_indicators)
        
        return has_positive and not has_negative


def demo_rephraser():
    """Demonstrate the IDK rephraser functionality"""
    print("=== IDK Rephraser Demo ===\n")
    
    rephraser = IDKRephraser(seed=42)
    
    # Show statistics
    stats = rephraser.get_statistics()
    print(f"📊 Total IDK templates available: {stats['total']}")
    for key, value in stats.items():
        if key != 'total':
            print(f"   - {key}: {value}")
    print()
    
    # Show samples from each style
    print("🎭 Sample IDK responses by style:")
    for style in IDKStyle:
        sample = rephraser.generate_random_idk(style)
        print(f"   {style.value.title()}: '{sample}'")
    print()
    
    # Show rephrasing functionality
    print("🔄 IDK Rephrasing Examples:")
    original_idks = ["I don't know", "I have no comment", "I'm not sure"]
    
    for original in original_idks:
        rephrased = rephraser.rephrase_idk_answer(original)
        print(f"   Original: '{original}'")
        print(f"   Rephrased: '{rephrased}'\n")
    
    # Show diverse set generation
    print("🔄 Diverse IDK set (10 variants):")
    diverse_set = rephraser.generate_diverse_set(10)
    for i, response in enumerate(diverse_set, 1):
        print(f"   {i:2d}. {response}")
    print()
    
    print("✅ IDK Rephraser Demo Complete!")


if __name__ == "__main__":
    demo_rephraser()