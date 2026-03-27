"""
Prompt templates for Garden Doctor application.
"""

from typing import List, Optional


class PromptTemplate:
    """Base class for prompt templates."""
    
    @staticmethod
    def format(**kwargs) -> str:
        """Format the prompt with given arguments."""
        raise NotImplementedError


class DiagnosisPrompt(PromptTemplate):
    """Prompt template for plant disease diagnosis."""
    
    TEMPLATE = """Analyze this plant leaf image for disease detection.

Instructions:
1. Identify the plant species if possible
2. Determine if the leaf is healthy or diseased
3. If diseased, identify the specific disease name
4. Provide a confidence level (High/Medium/Low)
5. Describe visible symptoms
6. Explain the cause of the condition
7. Provide treatment recommendations for {climate} climate
8. Suggest prevention measures

Please format your response as follows:
DISEASE: [disease name or "Healthy"]
CONFIDENCE: [High/Medium/Low]
SYMPTOMS: [description of visible symptoms]
CAUSE: [explanation of what causes this condition]
TREATMENT: [numbered list of treatment steps for {climate} climate]
PREVENTION: [bullet points for prevention tips]

Consider that this plant is growing in a {climate} climate zone when providing recommendations."""
    
    @staticmethod
    def format(climate: str = "Temperate") -> str:
        """
        Format the diagnosis prompt.
        
        Args:
            climate: Climate zone for tailored recommendations
            
        Returns:
            Formatted prompt string
        """
        return DiagnosisPrompt.TEMPLATE.format(climate=climate)


class QuickDiagnosisPrompt(PromptTemplate):
    """Shorter prompt for quick diagnosis."""
    
    TEMPLATE = """Identify any disease in this plant leaf image.
Provide: disease name, confidence (High/Medium/Low), brief treatment for {climate} climate.
Format: DISEASE: [name] | CONFIDENCE: [level] | TREATMENT: [brief steps]"""
    
    @staticmethod
    def format(climate: str = "Temperate") -> str:
        """Format the quick diagnosis prompt."""
        return QuickDiagnosisPrompt.TEMPLATE.format(climate=climate)


class DetailedAnalysisPrompt(PromptTemplate):
    """Detailed prompt for comprehensive analysis."""
    
    TEMPLATE = """You are an expert plant pathologist. Analyze this plant leaf image comprehensively.

Please provide a detailed analysis including:

## Identification
- Plant species (if identifiable)
- Disease or condition name
- Scientific name of pathogen (if applicable)
- Confidence level (High/Medium/Low with percentage estimate)

## Visual Analysis
- Describe all visible symptoms in detail
- Note color changes, spots, lesions, wilting, etc.
- Describe the pattern and distribution of symptoms
- Assess severity (mild/moderate/severe)

## Pathology
- Explain the biological cause of this condition
- Describe the disease progression
- List environmental conditions that favor this disease

## Treatment Plan for {climate} Climate
Provide specific, actionable treatment steps considering the local climate:

### Immediate Actions
1. [step 1]
2. [step 2]

### Chemical Treatments (if applicable)
- Product recommendations
- Application rates and timing

### Organic/Natural Alternatives
- Home remedies
- Organic-approved treatments

### Cultural Practices
- Watering adjustments
- Pruning recommendations
- Soil management

## Prevention Strategy
- Long-term prevention measures
- Crop rotation suggestions
- Resistant varieties to consider
- Monitoring recommendations

## Climate-Specific Advice
Additional considerations for growing in a {climate} climate zone."""
    
    @staticmethod
    def format(climate: str = "Temperate") -> str:
        """Format the detailed analysis prompt."""
        return DetailedAnalysisPrompt.TEMPLATE.format(climate=climate)


class ComparisonPrompt(PromptTemplate):
    """Prompt for comparing healthy vs diseased leaves."""
    
    TEMPLATE = """Compare these two plant leaf images.

Image 1 is a {label1} leaf.
Image 2 is a {label2} leaf.

Please analyze:
1. Key visual differences between the leaves
2. What symptoms indicate disease (if applicable)
3. How to distinguish healthy from diseased leaves of this plant type
4. Early warning signs to watch for

Provide a clear, educational comparison."""
    
    @staticmethod
    def format(label1: str, label2: str) -> str:
        """Format the comparison prompt."""
        return ComparisonPrompt.TEMPLATE.format(label1=label1, label2=label2)


def get_prompt_by_name(name: str) -> type:
    """
    Get a prompt class by name.
    
    Args:
        name: Prompt name (diagnosis, quick, detailed, comparison)
        
    Returns:
        Prompt class
        
    Raises:
        ValueError: If prompt name not found
    """
    prompts = {
        "diagnosis": DiagnosisPrompt,
        "quick": QuickDiagnosisPrompt,
        "detailed": DetailedAnalysisPrompt,
        "comparison": ComparisonPrompt,
    }
    
    if name.lower() not in prompts:
        raise ValueError(f"Unknown prompt: {name}. Available: {list(prompts.keys())}")
    
    return prompts[name.lower()]


def list_available_prompts() -> List[str]:
    """
    List all available prompt types.
    
    Returns:
        List of prompt names
    """
    return ["diagnosis", "quick", "detailed", "comparison"]
