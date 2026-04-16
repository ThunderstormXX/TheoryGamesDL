import re

with open("experiments/exp8/gpu_version/generate_section4_graphics.py", "r") as f:
    code = f.read()

# Fix the syntax error at the end of the file
code = code.replace("print(\"\\nAll experiments completed.\")", "print('\\nAll experiments completed.')")
code = code.replace("print(\"\nAll experiments completed.\")", "print('\\nAll experiments completed.')")

with open("experiments/exp8/gpu_version/generate_section4_graphics.py", "w") as f:
    f.write(code)

