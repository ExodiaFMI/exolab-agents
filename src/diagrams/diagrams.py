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

# --- Existing code for axodraw diagram generation ---

@dataclass
class AxodrawDiagramOutput:
    document_content: str
    diagram_width: float  # Width in points (pt)
    diagram_height: float  # Height in points (pt)

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

limiter = AsyncLimiter(max_rate=60, time_period=60)

async def generate_axodraw_diagram(prompt: str) -> AxodrawDiagramOutput:
    prompt_context = f"Prompt: {prompt}"
    async with limiter:
        result = await Runner.run(agent_axodraw, prompt_context)
    return result.final_output

class AxodrawDiagramRequest(BaseModel):
    prompt: str

    class Config:
        schema_extra = {
            "example": {
                "prompt": "Create a Feynman diagram representing electron-positron annihilation."
            }
        }

# Original endpoint (for reference)
router = APIRouter()

@router.post("/diagram/generate", response_model=dict,
             responses={
                 200: {
                     "content": {
                         "application/json": {
                             "example": {
                                 "document_content": "% LaTeX document content with axodraw2 diagram...",
                                 "diagram_width": 300,
                                 "diagram_height": 200
                             }
                         }
                     }
                 }
             })
async def generate_diagram(request: AxodrawDiagramRequest = Body(...)):
    try:
        diagram_output = await generate_axodraw_diagram(request.prompt)
        return {
            "document_content": diagram_output.document_content,
            "diagram_width": diagram_output.diagram_width,
            "diagram_height": diagram_output.diagram_height
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# --- Helper functions and constants for LaTeX to PNG conversion ---

# File and conversion settings
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

# --- New endpoint for generating PNG of the diagram ---

@router.post("/diagram/generate/png", response_class=FileResponse,
             responses={200: {
                 "content": {
                     "image/png": {
                         "example": "binary PNG data"
                     }
                 }
             }})
async def generate_diagram_png(request: AxodrawDiagramRequest = Body(...)):
    """
    Generate a LaTeX document with an axodraw2 diagram based on the provided prompt,
    compile it into a PDF, convert and crop the PDF to a PNG image, and return the PNG.
    """
    try:
        # 1. Use the existing agent to generate the LaTeX document and diagram dimensions.
        diagram_output = await generate_axodraw_diagram(request.prompt)
        
        # 2. Write the LaTeX content (which must include the diagram size comment)
        with open(TEX_FILE, "w") as f:
            f.write(diagram_output.document_content)
        
        # 3. Compile the LaTeX document to generate the PDF.
        compile_latex()
        
        # 4. Get the page dimensions from the generated PDF.
        page_width, _ = get_page_size(PDF_FILE)
        
        # 5. Retrieve the diagram dimensions from the TeX file.
        diag_width, diag_height = get_diagram_size(TEX_FILE)
        
        # 6. Convert the PDF to a cropped PNG using the helper function.
        convert_pdf_to_png(PDF_FILE, PNG_FILE, diag_width, diag_height, page_width)
        print(f"PNG created: {PNG_FILE}")
        
        # 7. Cleanup temporary files (leaving only the PNG).
        cleanup_temp_files()
        print("Temporary files removed. Only the PNG remains.")
        
        # 8. Return the PNG file as the response.
        return FileResponse(PNG_FILE, media_type="image/png", filename="diagram.png")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))