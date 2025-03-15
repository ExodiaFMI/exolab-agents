from fastapi import APIRouter, HTTPException, Body
from fastapi.responses import FileResponse
import subprocess
import re
import sys
import os
import asyncio
from dataclasses import dataclass
from agents import Agent, Runner, ModelSettings
from agents.agent_output import AgentOutputSchema
from pydantic import BaseModel
from aiolimiter import AsyncLimiter  # Rate limiter

# --- Dataclass for diagram output ---
@dataclass
class AxodrawDiagramOutput:
    document_content: str
    diagram_width: float  # Width in points (pt)
    diagram_height: float  # Height in points (pt)

# --- Original agent (for reference) ---
agent_axodraw = Agent(
    name="Axodraw Diagram Generator",
    output_type=AxodrawDiagramOutput,
    model="gpt-4o",
    model_settings=ModelSettings(temperature=1.3),
    instructions='''You are given a prompt related to physics, chemistry, or biology. Your task is to generate a complete LaTeX document that uses the axodraw2 package to create a diagram corresponding to the given prompt.

The document must include:
- A proper LaTeX preamble with the axodraw2 package.
- The diagram inside a picture environment using axodraw2 commands.
- A clearly defined diagram of an appropriate size.
- At the very end of the document, include a comment in the following format:
  %% Diagram Size: WIDTH x HEIGHT
  where WIDTH and HEIGHT are numerical values in points (pt).

Output your result as a JSON object with the following keys:
"document_content": string containing the entire LaTeX document,
"diagram_width": a number representing the width of the diagram in points,
"diagram_height": a number representing the height of the diagram in points.
Do not include any additional keys or commentary.'''
)

# --- New agent using o3-mini (plain text output) ---
agent_axodraw_o3 = Agent(
    name="Axodraw Diagram Generator (o3-mini)",
    output_type=str,  # Plain text output
    model="o3-mini",
    instructions='''You are given a prompt related to physics, chemistry, or biology. Your task is to generate a complete LaTeX document that uses the axodraw2 package to create a diagram corresponding to the given prompt.


The document must include:
- A proper LaTeX preamble with the axodraw2 package.
- The diagram inside a picture environment using axodraw2 commands.
- A clearly defined diagram of an appropriate size.
- At the very end of the document, include a comment in the following format:
  %% Diagram Size: WIDTH x HEIGHT
  where WIDTH and HEIGHT are numerical values in points (pt).

Output only the raw LaTeX code as plain text (do not output in JSON format).

BE CAREFULL: USE SPACE BETWEEN DIFFERENT ITEMS, BECUASE RIGHT NOW YOUR RESULTS ARE TOO CLOSE TO EACHOTHER

Here is documentation: Below is a more structured summary of the Axodraw documentation, organized by topic and function. This overview captures the abstract, usage instructions, command reference, and several illustrative examples.

⸻

Axodraw Documentation Summary

Axodraw is a LaTeX package that provides a set of PostScript drawing primitives. It is primarily used to draw Feynman diagrams, flow charts, and simple graphics—allowing whole articles (including pictures) to be exchanged in a single file. It relies on a PostScript interpreter (typically via dvips) and supports color (if the accompanying colordvi.sty is present).

⸻

1. Overview and Installation

Abstract
	•	Purpose:
	•	Draw Feynman diagrams, flow charts, and simple graphics using LaTeX.
	•	Integrate text and graphics in a single file.
	•	Key Feature:
	•	Uses PostScript for drawing commands.
	•	Color Support:
	•	Available if colordvi.sty is installed (default in most TeX distributions).
	•	Note:
	•	Earlier version was published in Comp. Phys. Comm. 83 (1994) 45.

Installation & Basic Setup
	•	Include Style File:
Insert the style file in the document preamble, e.g.:

\documentstyle[a4,11pt,axodraw]{article}


	•	Dependencies:
	•	The file epsf.sty (commonly available) is read by axodraw.sty.
	•	colordvi.sty is optional; if missing, color commands won’t be active.
	•	Compatibility:
	•	Designed to work with dvips (from Radical Eye Software).
	•	If using a different dvi-to-PostScript converter, syntax adjustments might be needed.

⸻

2. Using Axodraw in LaTeX
	•	Drawing Environment:
	•	Commands are executed inside a picture or figure environment.
	•	Coordinates are given in points (1 inch = 72 points).
	•	Scaling:
	•	Scale transformations are available via \SetScale{}.
	•	Note: Only PostScript text (using \PText) scales; regular LaTeX text does not.

⸻

3. Command Reference

The Axodraw package provides numerous commands. Below is a categorized overview:

3.1. Line and Arc Commands
	•	Straight Lines:
	•	\Line(x1,y1)(x2,y2)
Draws a straight line.
	•	\ArrowLine(x1,y1)(x2,y2)
Line with an arrow in the middle.
	•	\DashLine(x1,y1)(x2,y2){dashsize}
Dashed line with specified dash length.
	•	\LongArrow(x1,y1)(x2,y2)
Line with an arrow at the end.
	•	Arc Segments:
	•	\ArrowArc(x,y)(r,φ1,φ2)
Counterclockwise arc with an arrow in the middle.
	•	\ArrowArcn(x,y)(r,φ1,φ2)
Clockwise arc with an arrow in the middle.
	•	\LongArrowArc(x,y)(r,φ1,φ2) / \LongArrowArcn(x,y)(r,φ1,φ2)
Arc with an arrow at the end.
	•	\DashArrowArc(x,y)(r,φ1,φ2){dashsize} / \DashArrowArcn(x,y)(r,φ1,φ2){dashsize}
Dashed arc with an arrow.

3.2. Gluon and Photon Commands
	•	Gluons:
	•	\Gluon(x1,y1)(x2,y2){amplitude}{windings}
Draws a gluon line between two points.
The amplitude (can be negative) influences the side on which the curls appear.
	•	\GlueArc(x,y)(r,φ1,φ2){amplitude}{windings}
Gluon drawn along an arc segment.
	•	Photons:
	•	\Photon(x1,y1)(x2,y2){amplitude}{wiggles}
Photon line with wiggles between two points.
	•	\PhotonArc(x,y)(r,φ1,φ2){amplitude}{wiggles}
Photon drawn along an arc. For best symmetry, the number of wiggles is typically an integer plus 0.5.

3.3. Dashed and Zigzag Commands
	•	Dashed Curves:
	•	\DashCurve{(x1,y1)(x2,y2)...(xn,yn)}{dashsize}
Smooth dashed curve through a set of points.
	•	Zigzag Lines:
	•	\ZigZag(x1,y1)(x2,y2){amplitude}{wiggles}
Draws a zigzag line with specified amplitude and number of oscillations.

3.4. Box, Circle, and Oval Commands
	•	Blanked Boxes/Circles (overwrite existing content):
	•	\BBox(x1,y1)(x2,y2) and \BBoxc(x,y)(width,height)
	•	\BCirc(x,y){r}
	•	Colored Boxes/Circles:
	•	\CBox(x1,y1)(x2,y2){color1}{color2}
Box with border color color1 and background color2.
	•	\CCirc(x,y){r}{color1}{color2}
	•	\COval(x,y)(h,w)(φ){color1}{color2}
Oval with rotation φ.
	•	Gray Scale Boxes/Circles:
	•	\GBox(x1,y1)(x2,y2){grayscale}, \GBoxc(x,y)(width,height){grayscale}
	•	\GCirc(x,y){r}{grayscale}

3.5. Text Commands
	•	LaTeX Text:
	•	\Text(x,y)[mode]{text}
Places LaTeX text at a given focal point. Mode options (l, r, t, b, etc.) determine alignment.
	•	\rText(x,y)[mode][rotation]{text}
Rotated text (rotations: l = left 90°, r = right 90°, u = 180°).
	•	PostScript Text (scalable):
	•	\PText(x,y)(φ)[mode]{text}
PostScript text; affected by scaling and color commands.
	•	Text Boxes with Automatic Sizing:
	•	One-line: \BText(x,y){text} and \CText(x,y){color1}{color2}{text}
	•	Two-line: \B2Text(x,y){text1}{text2} and \C2Text(x,y){color1}{color2}{text1}{text2}

3.6. Additional Utility Commands
	•	Polygonal Shapes:
	•	\BTri(x1,y1)(x2,y2)(x3,y3) and \CTri(x1,y1)(x2,y2)(x3,y3){color1}{color2}
	•	Curves Through Points:
	•	\Curve{(x1,y1)(x2,y2)...(xn,yn)}
Fits a smooth curve (ensuring continuity in the first and second derivatives).
	•	Axis Drawing for Graphs:
	•	Linear Axis: \LinAxis(x1,y1)(x2,y2)(ND,d,hashsize,offset,width)
	•	Logarithmic Axis: \LogAxis(x1,y1)(x2,y2)(NL,hashsize,offset,width)
	•	Graphics Settings:
	•	\SetColor{NameOfColor} – sets drawing color.
	•	\SetPFont{fontname}{fontsize} – selects the PostScript font.
	•	\SetScale{scalevalue} – scales graphics (text in PText scales, regular LaTeX text does not).
	•	\SetOffset(x offset,y offset) – shifts coordinates at the LaTeX level.
	•	\SetScaledOffset(x offset,y offset) – shifts coordinates after scaling.
	•	\SetWidth{widthvalue} – sets line width.

3.7. Conditional Color Command
	•	\IfColor{arg1}{arg2}
Executes arg1 if the color package (colordvi.sty) is present; otherwise, executes arg2.

⸻

4. Examples and Use Cases

4.1. Text Modes

Illustrates the effect of different alignment options in text commands. For instance, placing text with positions:
	•	[lt] for left-top
	•	[l] for left-center
	•	[lb] for left-bottom
…and so on, with small circles marking the focal points.

4.2. Gluon Windings

Demonstrates gluon lines with different numbers of windings (curl density).
	•	Shows how varying the windings parameter (from 4 up to 8) alters the appearance.
	•	Also illustrates the effect of positive versus negative amplitude in determining on which side the curls appear.

4.3. Scaling

Explains how to use \SetScale{} to change the overall size of the drawing.
	•	Scaling may require adjustments to other parameters (e.g., amplitude or windings) to maintain visual appeal.
	•	Note: Only PostScript text (using \PText) scales; TeX text remains fixed.

4.4. Photons

Guidelines for drawing photons:
	•	Use an appropriate number of wiggles (often an integer plus 0.5) to achieve a symmetric appearance.
	•	The sign of the amplitude influences whether wiggles start “up” or “down.”

4.5. Flowcharts

Combines boxes with text, arrows, and curves to create flowcharts.
	•	Commands such as \BText, \C2Text, and various arrow commands are used.
	•	Example flowcharts may illustrate an automatic computation system for cross-sections.

4.6. Curves and Graphs

Integrates curve fitting with axis drawing:
	•	Uses \Curve for smooth interpolation between data points.
	•	Accompanies with linear and logarithmic axis commands (\LinAxis and \LogAxis) for complete graphing.
	•	Demonstrates labeling and arrow usage to annotate graphs (for example, representing threshold effects in particle production).

4.7. A Playful Example

A short example combining several elements (lines, gluons, photons, zigzags, and ovals) to showcase mixed usage in one picture.

⸻

5. Additional Information
	•	Acknowledgement:
The author thanks G.J. van Oldenborgh for assistance with some of the TeX macros.
	•	Obtaining Axodraw:
The package can be downloaded from the FORM homepage:
http://nikhef.nl/~form
Suggestions and commentary should be sent to t68@nikhef.nl.

⸻

This structured summary should help you navigate the various features and commands of Axodraw more easily while serving as a quick reference to its usage and examples.'''
)

limiter = AsyncLimiter(max_rate=60, time_period=60)

# --- Helper function: Parse plain text output into a JSON-like object ---
def parse_axodraw_output(output: str) -> AxodrawDiagramOutput:
    # Look for the diagram size comment in the output text.
    match = re.search(r"%%\s*Diagram Size:\s*([\d.]+)\s*x\s*([\d.]+)", output)
    if not match:
        raise ValueError("Diagram size comment not found in the output.")
    width = float(match.group(1))
    height = float(match.group(2))
    return AxodrawDiagramOutput(document_content=output, diagram_width=width, diagram_height=height)

# --- Async function using o3-mini model ---
async def generate_axodraw_diagram_o3(prompt: str) -> AxodrawDiagramOutput:
    prompt_context = f"Prompt: {prompt}"
    async with limiter:
        result = await Runner.run(agent_axodraw_o3, prompt_context)
    # result.final_output is plain text; parse it to extract dimensions.
    return parse_axodraw_output(result.final_output)

# --- Request model ---
class AxodrawDiagramRequest(BaseModel):
    prompt: str

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Create a Feynman diagram representing electron-positron annihilation."
            }
        }

router = APIRouter()

# --- Endpoint using the o3-mini–generated document ---
@router.post("/diagram/generate", response_model=dict,
             responses={200: {"content": {"application/json": {"example": {
                 "document_content": "% LaTeX document content...",
                 "diagram_width": 300,
                 "diagram_height": 200
             }}}}})
async def generate_diagram(request: AxodrawDiagramRequest = Body(...)):
    try:
        diagram_output = await generate_axodraw_diagram_o3(request.prompt)
        return {
            "document_content": diagram_output.document_content,
            "diagram_width": diagram_output.diagram_width,
            "diagram_height": diagram_output.diagram_height
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Helper functions and constants for LaTeX to PNG conversion ---

TEX_FILE = "mydiagram.tex"
PDF_FILE = "mydiagram.pdf"
PNG_FILE = "mydiagram.png"
DENSITY = 300  # DPI for conversion

def run_command(cmd, cwd=None):
    """Run a shell command and check for errors."""
    print(f"Running command: {' '.join(cmd)}")
    try:
        subprocess.run(cmd, check=True, cwd=cwd)
    except subprocess.CalledProcessError:
        print(f"Error running command: {' '.join(cmd)}")
        sys.exit(1)

def compile_latex():
    """Compile the LaTeX file (with axohelp step) to produce a PDF."""
    run_command(["pdflatex", TEX_FILE])
    run_command(["axohelp", os.path.splitext(TEX_FILE)[0]])
    run_command(["pdflatex", TEX_FILE])

def get_page_size(pdf_file):
    """Extract page dimensions from the PDF using pdfinfo."""
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
    # Adjusting the dimensions if needed (here adding 200 to each)
    diag_width = float(match.group(1)) + 200
    diag_height = float(match.group(2)) + 200
    print(f"Diagram size (from .tex): {diag_width} x {diag_height} pts")
    return diag_width, diag_height

def convert_pdf_to_png(pdf_file, png_file, diag_width, diag_height, page_width):
    """
    Convert the PDF to PNG and crop it to the diagram.
    Assumes the diagram is centered horizontally at the top of the page.
    """
    factor = DENSITY / 72.0  # Conversion: 1 pt = 1/72 inch.
    crop_width = int(round(diag_width * factor))
    crop_height = int(round(diag_height * factor))
    page_width_px = int(round(page_width * factor))
    left_offset = int(round((page_width_px - crop_width) / 2))
    top_offset = 0
    crop_geometry = f"{crop_width}x{crop_height}+{left_offset}+{top_offset}"
    print(f"Cropping with geometry: {crop_geometry}")

    full_png = "full_temp.png"
    run_command([
        "gs", "-q", "-dNOPAUSE", "-dBATCH",
        "-sDEVICE=pngalpha",
        f"-r{DENSITY}",
        "-sOutputFile=" + full_png,
        pdf_file
    ])
    run_command([
        "convert",
        full_png,
        "-crop", crop_geometry,
        "+repage",
        png_file
    ])
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

@router.post("/diagram/generate/png", response_class=FileResponse,
             responses={200: {"content": {"image/png": {"example": "binary PNG data"}}}})
async def generate_diagram_png(request: AxodrawDiagramRequest = Body(...)):
    """
    Generate a LaTeX document with an axodraw2 diagram using the o3-mini model,
    compile it into a PDF, convert and crop the PDF to a PNG image, and return the PNG.
    """
    try:
        # 1. Use the o3-mini agent to generate the LaTeX document.
        diagram_output = await generate_axodraw_diagram_o3(request.prompt)
        
        # 2. Write the LaTeX content to file.
        with open(TEX_FILE, "w") as f:
            f.write(diagram_output.document_content)
        
        # 3. Compile the LaTeX document.
        compile_latex()
        
        # 4. Get the page dimensions from the generated PDF.
        page_width, _ = get_page_size(PDF_FILE)
        
        # 5. Retrieve the diagram dimensions from the TeX file.
        diag_width, diag_height = get_diagram_size(TEX_FILE)
        
        # 6. Convert the PDF to a cropped PNG.
        convert_pdf_to_png(PDF_FILE, PNG_FILE, diag_width, diag_height, page_width)
        print(f"PNG created: {PNG_FILE}")
        
        # 7. Cleanup temporary files (leaving only the PNG).
        cleanup_temp_files()
        print("Temporary files removed. Only the PNG remains.")
        
        # 8. Return the PNG file as the response.
        return FileResponse(PNG_FILE, media_type="image/png", filename="diagram.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))