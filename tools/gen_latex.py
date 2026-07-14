import random
import matplotlib.pyplot as plt
import os
from typing import List

# --- Latex Generator ---
class Ops:
    def __init__(self, numberOfOperated: int):
        self.innerOps: List['Ops'] = []
        self.numberOfOperated = numberOfOperated
        if self.numberOfOperated == 0:
            characters = [
                # Biến thường
                "x", "y", "z", "a", "b", "c", "d", "m", "n", "t", "s", "u", "v", "w",
                # Số
                "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
                # Hằng số toán học
                "e", "i", "\\pi", "\\phi", "\\infty",
                # Ký tự đặc biệt
                # Ký tự LaTeX phổ biến
                "\\alpha", "\\beta", "\\gamma", "\\delta", "\\epsilon", "\\theta", "\\lambda", "\\mu", "\\rho", "\\sigma", "\\tau", "\\omega",
                # Ký tự logic
                # "\\top", "\\bot", "\\forall", "\\exists", "\\neg", "\\land", "\\lor", "\\to", "\\implies", "\\iff"
            ]
        elif self.numberOfOperated == 1:
            characters = [
                "+{{{}}}", "-{{{}}}", "\\sin{{{}}}", "\\cos{{{}}}", "\\tan{{{}}}", "\\log{{{}}}",
                "\\ln{{{}}}", "\\exp{{{}}}", "\\det{{{}}}", "\\sqrt{{{}}}",
                # Chỉ đóng mở ngoặc tròn
                "({})"
            ]
        elif self.numberOfOperated == 2:
            characters = [
                "{{{}}}+{{{}}}", "{{{}}}-{{{}}}", "{{{}}}\\times{{{}}}", "{{{}}}\\cdot{{{}}}",
                "\\frac{{{}}}{{{}}}", "\\binom{{{}}}{{{}}}", "{{{}}}^{{{}}}", "{{{}}}_{{{}}}",
                # Chỉ đóng mở ngoặc tròn cho biểu thức 2 ngôi
                "({}+{})", "({}-{})"
            ]
        elif self.numberOfOperated == 3:
            characters = [
                "\\int_{{{}}}^{{{}}}{{{}}}\\,dx",
                "\\sum_{{{}}}^{{{}}}{{{}}}",
                "\\prod_{{{}}}^{{{}}}{{{}}}"
            ]
        else:
            characters = ["x"]
        self.character = random.choice(characters)

    def getInnerOps(self, max_depth: int = 4):
        if self.numberOfOperated <= 0:
            return
        weights = [0.6, 0.15, 0.2, 0.05]
        for _ in range(self.numberOfOperated):
            if max_depth <= 0:
                child = Ops(0)
            else:
                n = random.choices([0, 1, 2, 3], weights=weights)[0]
                child = Ops(n)
                child.getInnerOps(max_depth - 1)
            self.innerOps.append(child)

    def getLatex(self):
        if self.numberOfOperated == 0:
            return self.character
        return self.character.format(*[child.getLatex() for child in self.innerOps])

# --- Latex to Image ---
def latex_to_image(latex_code: str, filename="output.png"):
    fig, ax = plt.subplots()
    ax.axis("off")
    ax.text(0.5, 0.5, f"${latex_code}$", fontsize=25, ha="center", va="center")
    plt.savefig(filename, bbox_inches="tight", pad_inches=0.3, dpi=300)
    plt.close()

# --- Generate N latex images ---
def generate_latex_images(n=10, out_dir="Data/GenData"):
    os.makedirs(out_dir, exist_ok=True)
    records = []
    for i in range(n):
        root = Ops(random.randint(2, 3))
        root.getInnerOps(max_depth=3)
        latex = root.getLatex()
        file_name = os.path.join(out_dir, f"{i:08d}.png")
        latex_to_image(latex, file_name)
        records.append({"id": i, "file_name": file_name, "latex": latex})
    return records

if __name__ == "__main__":
    # Example: generate 20 latex images
    generate_latex_images(20)
