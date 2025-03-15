#!/usr/bin/env python3
import subprocess
import re
import sys
import os

# Constants
TEX_FILE = "mydiagram.tex"
PDF_FILE = "mydiagram.pdf"
PNG_FILE = "mydiagram.png"
DENSITY = 300  # DPI for conversion

# Example LaTeX document content.
# Note: The document must include a comment line with the diagram size.
LATEX_CONTENT = r"""
\documentclass{article}
\usepackage{axodraw2}
\begin{document}
\begin{center}
\begin{picture}(300,56)(0,0)
  \SetColor{Blue}
  \Line(100,25)(150,25)
  \SetColor{Green}
  \Gluon(150,25)(200,25){3}{6}
  \SetColor{Red}
  \Photon(150,35)(200,45){3}{6}
  \SetColor{Mahogany}
  \ZigZag(150,15)(200,5){3}{6}
  \IfColor
    {\COval(150,25)(20,10)(0){Black}{Yellow}}
    {\GOval(150,25)(20,10)(0){0.5}}
\end{picture}
\end{center}
%% Diagram Size: 150 x 250
\end{document}
"""

def run_command(cmd, cwd=None):
    """Run a shell command and check for errors."""
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
    except subprocess.CalledProcessError as e:
        print(f"Error running command: {' '.join(cmd)}")
        sys.exit(1)

def compile_latex():
    """Compile the LaTeX file (with axohelp step) to produce a PDF."""
    # First pdflatex run
    run_command(["pdflatex", TEX_FILE])
    # Run axohelp to process axodraw2 auxiliary files.
    run_command(["axohelp", os.path.splitext(TEX_FILE)[0]])
    # Second pdflatex run to incorporate axohelp changes.
    run_command(["pdflatex", TEX_FILE])

def get_page_size(pdf_file):
    """Use pdfinfo to extract page width and height (in points) from the PDF."""
    try:
        result = subprocess.run(["pdfinfo", pdf_file], capture_output=True, text=True, check=True)
    except subprocess.CalledProcessError:
        print("Error: pdfinfo failed to run.")
        sys.exit(1)
    match = re.search(r"Page size:\s+([\d.]+)\s+x\s+([\d.]+)\s+pts", result.stdout)
    if not match:
        print("Error: Could not parse page size from pdfinfo output.")
        sys.exit(1)
    page_width = float(match.group(1))
    page_height = float(match.group(2))
    print(f"Page size: {page_width} x {page_height} pts")
    return page_width, page_height

def get_diagram_size(tex_file):
    """Parse the LaTeX file to find the diagram size comment."""
    with open(tex_file, "r") as f:
        content = f.read()
    match = re.search(r"%%\s*Diagram Size:\s*([\d.]+)\s*x\s*([\d.]+)", content)
    if not match:
        print("Error: Diagram size comment not found in the TeX file.")
        sys.exit(1)
    diag_width = float(match.group(1))
    diag_height = float(match.group(2))
    print(f"Diagram size (from .tex): {diag_width} x {diag_height} pts")
    return diag_width, diag_height

def convert_pdf_to_png(pdf_file, png_file, diag_width, diag_height, page_width):
    """
    Convert the PDF to PNG and crop it to the diagram.
    Assumes the diagram is centered horizontally at the top of the page.
    This version uses Ghostscript to convert the PDF to PNG first,
    then crops the PNG with ImageMagick's convert.
    """
    # Conversion factor: points to pixels at given DPI (1 pt = 1/72 inch)
    factor = DENSITY / 72.0
    crop_width = int(round(diag_width * factor))
    crop_height = int(round(diag_height * factor))
    page_width_px = int(round(page_width * factor))
    # Diagram is centered horizontally; assume it starts at the top (y offset 0).
    left_offset = int(round((page_width_px - crop_width) / 2))
    top_offset = 0

    crop_geometry = f"{crop_width}x{crop_height}+{left_offset}+{top_offset}"
    print(f"Cropping with geometry: {crop_geometry}")

    # First, convert the full PDF page to a PNG using Ghostscript.
    full_png = "full_temp.png"
    run_command([
        "gs", "-q", "-dNOPAUSE", "-dBATCH",
        "-sDEVICE=pngalpha",
        f"-r{DENSITY}",
        "-sOutputFile=" + full_png,
        pdf_file
    ])
    
    # Now, crop the full PNG image to the diagram area.
    run_command([
        "convert",
        full_png,
        "-crop", crop_geometry,
        "+repage",
        png_file
    ])
    
    # Remove the temporary full-page PNG.
    os.remove(full_png)

def cleanup_temp_files():
    """Remove temporary files, leaving only the PNG."""
    temp_files = [
        "mydiagram.aux",
        "mydiagram.ax1",
        "mydiagram.ax2",
        "mydiagram.log",
        "mydiagram.pdf",
        "mydiagram.tex"
    ]
    for file in temp_files:
        if os.path.exists(file):
            print(f"Removing {file}")
            os.remove(file)

def main():
    # Write the LaTeX content to file.
    print(f"Writing LaTeX file: {TEX_FILE}")
    with open(TEX_FILE, "w") as f:
        f.write(LATEX_CONTENT)
    
    # Compile the LaTeX document.
    compile_latex()
    
    # Get page dimensions from PDF.
    page_width, page_height = get_page_size(PDF_FILE)
    # Get diagram size from the TeX file.
    diag_width, diag_height = get_diagram_size(TEX_FILE)
    
    # Convert the PDF to a cropped PNG.
    convert_pdf_to_png(PDF_FILE, PNG_FILE, diag_width, diag_height, page_width)
    print(f"PNG created: {PNG_FILE}")
    
    # Cleanup temporary files.
    cleanup_temp_files()
    print("Temporary files removed. Only the PNG remains.")

if __name__ == "__main__":
    main()