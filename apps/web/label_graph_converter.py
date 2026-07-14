"""
Label Graph to LaTeX Converter
Converts CROHME label graph format to LaTeX notation
"""

class Stack:
    """Simple stack implementation for managing LaTeX braces"""
    def __init__(self):
        self.stack = []

    def push(self, element):
        self.stack.append(element)

    def pop(self):
        if self.isEmpty():
            return None
        return self.stack.pop()

    def peek(self):
        if self.isEmpty():
            return None
        return self.stack[-1]

    def isEmpty(self):
        return len(self.stack) == 0

    def size(self):
        return len(self.stack)


def label_to_latex(label_graph: str) -> str:
    """
    Convert label graph format to LaTeX notation.
    
    Label graph format uses spatial relationships between symbols:
    - Right: horizontal spacing
    - Sub: subscript
    - Sup: superscript
    - Below: below relationship (for \sum, \int, fractions)
    - Above: above relationship (for fractions)
    - Inside: inside relationship (for \sqrt)
    - NoRel: no relation (context separator, closes blocks)
    
    Examples:
        "a Sub b" -> "a_{b}"
        "x Sup 2" -> "x^{2}"
        "2 NoRel - Below 3" -> "\\frac{2}{3}"
        "\\sum Below i Right = Right 1" -> "\\sum_{i} = 1"
    
    Args:
        label_graph: String in label graph format
        
    Returns:
        LaTeX string
    """
    relationship_vocabs = ["Right", "NoRel", "Sup", "Sub", "Below", "Inside", "Above", "COMMA"]
    function_vocabs = ["\\sqrt", "\\sin", "\\sum", "\\int", "\\cos", "\\log", "\\lim", "\\tan", 
                       "\\frac", "\\alpha", "\\beta", "\\gamma", "\\delta", "\\theta", "\\pi",
                       "\\sigma", "\\phi", "\\omega", "\\infty", "\\rightarrow", "\\leftarrow",
                       "\\leq", "\\geq", "\\neq", "\\times", "\\div", "\\pm", "\\mp"]
    
    labels = label_graph.split()
    latex = ""
    stack = Stack()
    i = 0
    
    # Pre-process to detect fraction patterns: "num NoRel - Below denom"
    # Replace with special markers
    processed_labels = []
    j = 0
    while j < len(labels):
        # Look ahead for fraction pattern: "num NoRel - Below denom"
        if (j + 3 < len(labels) and 
            labels[j + 1] == "NoRel" and 
            labels[j + 2] == "-" and 
            labels[j + 3] == "Below" and
            labels[j] not in relationship_vocabs):
            # Found fraction: num NoRel - Below denom
            numerator = labels[j]
            # Find denominator (next non-relationship token after Below)
            if j + 4 < len(labels) and labels[j + 4] not in relationship_vocabs:
                denominator = labels[j + 4]
                processed_labels.append(f"FRAC_START")
                processed_labels.append(numerator)
                processed_labels.append(f"FRAC_MID")
                processed_labels.append(denominator)
                processed_labels.append(f"FRAC_END")
                j += 5  # Skip num, NoRel, -, Below, denom
            else:
                # No denominator found, keep original
                processed_labels.append(labels[j])
                j += 1
        else:
            processed_labels.append(labels[j])
            j += 1
    
    # Now process the pre-processed labels
    i = 0
    while i < len(processed_labels):
        label = processed_labels[i]
        
        if label == "FRAC_START":
            latex += "\\frac{"
        elif label == "FRAC_MID":
            latex += "}{"
        elif label == "FRAC_END":
            latex += "}"
            
        elif label == "Right":
            # Horizontal spacing
            latex += " "
            
        elif label == "Sub":
            # Always use braces for subscripts
            latex += "_{"
            stack.push("Sub")
                
        elif label == "Sup":
            # Always use braces for superscripts
            latex += "^{"
            stack.push("Sup")
                
        elif label == "COMMA":
            latex += ","
            
        elif label == "Above":
            # Fraction numerator
            latex += "\\frac{"
            stack.push("\\frac")
                
        elif label == "Inside":
            # For \sqrt or other containers
            latex += "{"
            stack.push("Inside")
            
        elif label == "Below":
            # Subscript for \sum, \int, \lim
            if i > 0 and processed_labels[i - 1] in ["\\sum", "\\int", "\\lim"]:
                latex += "_{"
                stack.push("Below")
            else:
                # Standalone Below - might be part of pattern we missed
                latex += "_{"
                stack.push("Below")
                
        elif label == "NoRel":
            # Context separator: closes blocks or acts as separator
            current = stack.peek()
            
            if current == "\\frac":
                # Middle of fraction: close numerator, open denominator
                latex += "}{"
                stack.pop()
                stack.push("NoRel")
            elif current in ["Below", "Inside", "Sub", "Sup"]:
                # Close current block
                latex += "}"
                stack.pop()
            elif current == "NoRel":
                # Close denominator of fraction
                latex += "}"
                stack.pop()
            else:
                # Just spacing
                latex += " "
                
        elif label == "\\sqrt":
            latex += "\\sqrt"
            
        elif label == "\\sum":
            latex += "\\sum"
            
        elif label == "\\int":
            latex += "\\int"
            
        elif label == "\\lim":
            latex += "\\lim"
            
        elif label == "\\frac":
            latex += "\\frac"
            
        elif label in ["{", "}"]:
            # Direct braces - skip them as we handle braces explicitly
            pass
            
        elif label in function_vocabs:
            # Other LaTeX commands
            latex += label
            
        elif label not in relationship_vocabs and not label.startswith("FRAC_"):
            # Regular symbols, numbers, letters
            latex += label
            
        i += 1
    
    # Close any remaining open braces
    while not stack.isEmpty():
        latex += "}"
        stack.pop()
    
    return latex


if __name__ == "__main__":
    # Test cases
    test_cases = [
        ("a Sub b NoRel + Right b Sup c", "a_{b} + b^{c}"),
        ("x Sup 2 NoRel + Right y Sup 2", "x^{2} + y^{2}"),
        ("2 NoRel - Below 3", "\\frac{2}{3}"),
        ("2 NoRel - Below 3 NoRel + NoRel 1 NoRel - Below 2", "\\frac{2}{3} + \\frac{1}{2}"),
        ("\\sum Below i Right = Right 1", "\\sum_{i} = 1"),
        ("\\sqrt Inside x Sup 2 NoRel + Right y Sup 2", "\\sqrt{x^{2} + y^{2}}"),
        ("\\int Right d Right x", "\\int d x"),
        ("a Sub i NoRel j", "a_{ij}"),
    ]
    
    print("Testing label_to_latex function:")
    print("=" * 80)
    for label_graph, expected in test_cases:
        result = label_to_latex(label_graph)
        status = "✅ PASS" if result == expected else "❌ FAIL"
        print(f"\n{status}")
        print(f"Input:    {label_graph}")
        print(f"Expected: {expected}")
        print(f"Got:      {result}")
    print("\n" + "=" * 80)
