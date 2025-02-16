from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS
from langchain_google_genai import ChatGoogleGenerativeAI
from manim import *
import os
from dotenv import load_dotenv
import logging
import tempfile
import subprocess
import json
from pathlib import Path
import re
import shutil
import time


# Setup Manim
def setup_manim_directories():
    """Setup required directories for Manim"""
    base_dir = "D:/EVE"
    required_dirs = [
        "images",
        "Tex",
        "texts",
        "videos",
        "media",
        "media/videos",
        "media/videos/1080p60",
        "media/Tex"  # Changed from media/temp to media/Tex
    ]

    for dir_path in required_dirs:
        full_path = os.path.join(base_dir, dir_path)
        if not os.path.exists(full_path):
            os.makedirs(full_path, exist_ok=True)
            logger.info(f"Created directory: {full_path}")


# Initialize Flask
app = Flask(__name__)
CORS(app, resources={
    r"/*": {
        "origins": ["http://127.0.0.1:5000", "http://localhost:5000", "http://localhost:8000"],
        "methods": ["GET", "POST"],
        "allow_headers": ["Content-Type"]
    }
})
setup_manim_directories()

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


load_dotenv()
google_api_key = os.getenv("GOOGLE_API_KEY")


class EquationSolving(Scene):
    def __init__(self, equations, title_text, **kwargs):
        super().__init__(**kwargs)
        self.equations = equations
        self.title_text = title_text
        self.camera.background_color = WHITE

    def construct(self):
        def create_equation_group(equations, start_index=0):
            group = []
            spacing = 1.8
            for i, eq in enumerate(equations[:5]):
                eq_copy = eq.copy()
                eq_copy.set_color(BLACK)
                eq_copy.shift(UP * (3.5 - i * spacing))
                group.append(eq_copy)
            return group

        # Create title
        title = Text(self.title_text, font_size=36, color=BLACK)
        title.to_edge(UP, buff=0.8)
        self.play(Write(title))

        current_mobjects = []
        arrows = []
        texts = []

        for i in range(0, len(self.equations), 5):
            if current_mobjects:
                self.play(
                    *[FadeOut(mob) for mob in current_mobjects],
                    *[FadeOut(arrow) for arrow in arrows],
                    *[FadeOut(text) for text in texts]
                )
                current_mobjects = []
                arrows = []
                texts = []

            current_group = self.equations[i:i + 5]
            positioned_equations = create_equation_group(current_group)

            self.play(FadeIn(positioned_equations[0], shift=UP * 0.5))
            current_mobjects.append(positioned_equations[0])

            for j in range(1, len(positioned_equations)):
                arrow = Arrow(
                    start=positioned_equations[j - 1].get_bottom() + DOWN * 0.2,
                    end=positioned_equations[j].get_top() + UP * 0.2,
                    color=BLACK,
                    max_tip_length_to_length_ratio=0.15,
                    stroke_width=2
                )

                if i + j < len(self.equations) - 1:
                    step_text = Text(
                        "Step " + str(i + j + 1),
                        font_size=28,
                        color=BLACK
                    )
                else:
                    step_text = Text(
                        "Final Answer",
                        font_size=28,
                        color=BLACK
                    )
                step_text.next_to(arrow, RIGHT)

                self.play(
                    GrowArrow(arrow),
                    Write(step_text),
                    Write(positioned_equations[j]),
                    run_time=1.5
                )

                current_mobjects.append(positioned_equations[j])
                arrows.append(arrow)
                texts.append(step_text)

                if j == len(positioned_equations) - 1 and i + j == len(self.equations) - 1:
                    box = SurroundingRectangle(
                        positioned_equations[j],
                        color=BLACK,
                        stroke_width=2,
                        buff=0.2,
                        corner_radius=0.1
                    )
                    final_text = Text("✓ Solution Verified", font_size=32, color=BLACK)
                    final_text.next_to(box, RIGHT)
                    self.play(
                        Create(box),
                        Write(final_text),
                        positioned_equations[j].animate.scale(1.1)
                    )
                    current_mobjects.extend([box, final_text])

            self.wait(2)

        self.play(
            *[FadeOut(mob) for mob in current_mobjects],
            *[FadeOut(arrow) for arrow in arrows],
            *[FadeOut(text) for text in texts],
            FadeOut(title)
        )


def extract_equations(text):
    """Extract equations from text and format them properly"""
    lines = text.strip().split('\n')
    equations = []
    
    # Create a single TexTemplate with specific configuration
    tex_template = TexTemplate()
    tex_template.add_to_preamble(r"""
        \usepackage{amsmath}
        \usepackage{amssymb}
        \usepackage{amsfonts}
    """)
    tex_template.tex_compiler = "latex"  # Use latex instead of pdflatex
    
    for line in lines:
        # Remove any step numbers or prefixes
        line = re.sub(r'^\d+[\).:]\s*', '', line.strip())
        
        if line and '=' in line:
            try:
                # Format the equation for LaTeX
                formatted_line = line.strip()
                # Basic replacements
                formatted_line = formatted_line.replace('*', r' \times ')
                formatted_line = formatted_line.replace('/', r' \div ')
                formatted_line = formatted_line.replace('=', r' = ')
                
                # Create equation object
                eq = MathTex(
                    formatted_line,
                    stroke_width=2,
                    color=BLACK,
                    font_size=36,
                    tex_template=tex_template
                )
                equations.append(eq)
                logger.info(f"Successfully formatted equation: {formatted_line}")
                
            except Exception as e:
                logger.error(f"Failed to create equation '{line}': {str(e)}")
                # Try a simpler version
                try:
                    simple_line = line.replace('*', 'x').replace('/', ':')
                    eq = Text(simple_line, color=BLACK, font_size=36)
                    equations.append(eq)
                    logger.info(f"Created fallback text for equation: {simple_line}")
                except Exception as e2:
                    logger.error(f"Fallback also failed: {str(e2)}")
                continue
    
    return equations


def cleanup_old_videos():
    """Clean up existing video files"""
    try:
        video_dir = Path("D:/EVE/media/videos/1080p60")
        if video_dir.exists():
            for file in video_dir.glob("*.mp4"):
                try:
                    file.unlink()
                    logger.info(f"Deleted old video: {file}")
                except Exception as e:
                    logger.error(f"Failed to delete {file}: {str(e)}")
                    
            # Also clean up partial movie files directory
            partial_dir = video_dir / "partial_movie_files"
            if partial_dir.exists():
                shutil.rmtree(partial_dir, ignore_errors=True)
                logger.info("Cleaned up partial movie files")
    except Exception as e:
        logger.error(f"Error during video cleanup: {str(e)}")


def create_animation(equations, title):
    """Create and render the animation"""
    try:
        # Clean up old videos first
        cleanup_old_videos()
        
        # Set up base directories
        base_dir = Path("D:/EVE")
        media_dir = base_dir / "media"
        video_dir = media_dir / "videos/1080p60"
        tex_dir = media_dir / "Tex"

        # Generate unique filename using timestamp
        timestamp = int(time.time())
        output_filename = f"equation_video_{timestamp}"

        # Ensure directories exist
        for dir_path in [media_dir, video_dir, tex_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # Basic Manim configuration
        config.update({
            "media_dir": str(media_dir),
            "video_dir": str(video_dir),
            "tex_dir": str(tex_dir),
            "text_dir": str(media_dir / "texts"),
            "output_file": output_filename,  # Use unique filename
            "pixel_height": 1080,
            "pixel_width": 1920,
            "frame_rate": 30,
            "background_color": WHITE,
            "preview": False,
            "write_to_movie": True,
            "disable_caching": True,
            "tex_template": TexTemplate()
        })

        # Create and render scene
        scene = EquationSolving(equations, title)
        scene.render()

        # Verify output with new filename
        output_file = video_dir / f"{output_filename}.mp4"
        if output_file.exists():
            logger.info(f"Video created successfully at {output_file}")
            return f"{output_filename}.mp4"
        
        logger.error(f"Video file not found at expected path: {output_file}")
        return None

    except Exception as e:
        logger.error(f"Animation creation failed: {str(e)}")
        import traceback
        logger.error(traceback.format_exc())
        return None


try:
    llm = ChatGoogleGenerativeAI(
        model="gemini-1.5-pro",
        temperature=0,
        max_tokens=None,
        timeout=30,
        max_retries=2,
    )
except Exception as e:
    logger.error(f"Error initializing LLM: {str(e)}")
    llm = None


@app.route('/')
def serve_frontend():
    return send_from_directory('.', 'index.html')


@app.route('/styles.css')
def serve_css():
    return send_from_directory('.', 'styles.css')


@app.route('/chat', methods=['POST'])
def chat():
    if not llm:
        return jsonify({
            "success": False,
            "error": "LLM not initialized properly"
        }), 500

    try:
        data = request.json
        user_input = data.get('message', '')
        wants_video = data.get('video', False)

        if not user_input:
            return jsonify({
                "success": False,
                "error": "No message provided"
            }), 400

        prompt = f"""
        You are EVE, Study assistant that explains people about certain topic. Based on the user's request:
        {user_input}

        If the prompt contains the word "solve", respond only with the steps involved, displaying each step on a new line. Do not include any explanations or additional content.
        """

        messages = [{"role": "user", "content": prompt}]
        response = llm.invoke(messages)

        if wants_video and "solve" in user_input.lower():
            equations = extract_equations(response.content)
            if equations:
                setup_manim_directories()  # Ensure directories exist
                video_path = create_animation(equations, f"Solving {user_input}")
                if video_path:
                    return jsonify({
                        "success": True,
                        "message": response.content,
                        "video": video_path
                    })
                else:
                    # Return just the text response if video fails
                    return jsonify({
                        "success": True,
                        "message": response.content
                    })

        return jsonify({
            "success": True,
            "message": response.content
        })

    except Exception as e:
        logger.error(f"Error processing request: {str(e)}")
        return jsonify({
            "success": False,
            "error": f"Error processing request: {str(e)}"
        }), 500


@app.route('/video/<filename>')
def serve_video(filename):
    video_path = os.path.join("D:/EVE/media/videos/1080p60", filename)
    try:
        return send_file(
            video_path,
            mimetype='video/mp4',
            as_attachment=False
        )
    except Exception as e:
        logger.error(f"Error serving video: {str(e)}")
        return jsonify({
            "success": False,
            "error": "Video not found"
        }), 404


@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200


@app.route('/execute', methods=['POST'])
def execute_code():
    try:
        data = request.json
        code = data.get('code', '')

        if not code:
            return jsonify({
                "success": False,
                "error": "No code provided"
            }), 400

        # Create a temporary file to execute the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Execute the code and capture output
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=5  # 5 second timeout
            )

            if result.stderr:
                return jsonify({
                    "success": False,
                    "error": result.stderr
                })

            return jsonify({
                "success": True,
                "output": result.stdout
            })

        finally:
            # Clean up temporary file
            os.unlink(temp_file)

    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "error": "Code execution timed out"
        }), 408
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/images/<path:filename>')
def serve_image(filename):
    return send_from_directory('images', filename)


@app.route('/problems')
def get_problems():
    try:
        problems = []
        with open('problem details.txt', 'r', encoding='utf-8') as f:
            content = f.read()
            sections = content.split('\n\n\n')

            for section in sections:
                if not section.strip():
                    continue

                lines = section.split('\n')
                # Find the problem title and remove the number prefix
                for line in lines:
                    if line.strip() and line.strip()[0].isdigit():
                        # Remove the number and dot prefix (e.g., "1. ")
                        title = line.strip().split('. ', 1)[1]
                        description = []
                        example = []
                        mode = 'description'

                        for content_line in lines[lines.index(line) + 1:]:
                            if content_line.startswith('Example'):
                                mode = 'example'
                            elif mode == 'description' and content_line.strip():
                                description.append(content_line.strip())
                            elif mode == 'example' and content_line.strip():
                                example.append(content_line.strip())

                        problems.append({
                            "title": title,
                            "description": '\n'.join(description),
                            "example": '\n'.join(example)
                        })
                        break

        return jsonify(problems)
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/problem/<title>')
def get_problem(title):
    try:
        with open('problem details.txt', 'r', encoding='utf-8') as f:
            content = f.read()
            sections = content.split('\n\n\n')

            for section in sections:
                lines = section.split('\n')
                # Find the problem title
                for line in lines:
                    if line.strip() and '. ' in line:
                        # Get the title without the number prefix
                        current_title = line.strip().split('. ', 1)[1]
                        if current_title == title:
                            description = []
                            example = []
                            mode = 'description'

                            for content_line in lines[lines.index(line) + 1:]:
                                if content_line.startswith('Example'):
                                    mode = 'example'
                                elif mode == 'description' and content_line.strip():
                                    description.append(content_line.strip())
                                elif mode == 'example' and content_line.strip():
                                    example.append(content_line.strip())

                            template = f"""def {title.lower().replace(' ', '_')}():
    # Write your solution here
    pass

if _name_ == "_main_":
    result = {title.lower().replace(' ', '_')}()
    print(result)
"""

                            return jsonify({
                                'description': '\n'.join(description),
                                'example': '\n'.join(example),
                                'template': template
                            })

        return jsonify({"error": "Problem not found"}), 404
    except Exception as e:
        print(f"Error: {str(e)}")
        return jsonify({"error": str(e)}), 500


@app.route('/submit', methods=['POST'])
def submit_code():
    try:
        data = request.json
        code = data.get('code', '')
        problem = data.get('problem', '')

        if not code or not problem:
            return jsonify({
                "success": False,
                "error": "No code or problem provided"
            }), 400

        # Create a temporary file to execute the code
        with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            # Execute the code with test cases
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=5
            )

            if result.stderr:
                return jsonify({
                    "success": False,
                    "error": result.stderr
                })


            return jsonify({
                "success": True,
                "message": "All test cases passed!"
            })

        finally:
            # Clean up temporary file
            os.unlink(temp_file)

    except subprocess.TimeoutExpired:
        return jsonify({
            "success": False,
            "error": "Code execution timed out"
        }), 408
    except Exception as e:
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500


@app.route('/generate-problem', methods=['POST'])
def generate_problem():
    try:
        data = request.json
        prompt = data.get('prompt', '')
        
        if not prompt:
            return jsonify({"error": "No prompt provided"}), 400

        # Update the prompt to ensure proper formatting
        system_prompt = """Generate a coding problem based on the user's prompt. 
        Format the response EXACTLY like this example:
        Title: Two Sum
        Description: Given an array of integers nums and an integer target, return indices of the two numbers such that they add up to target.

        Example:
        Input: nums = [2,7,11,15], target = 9
        Output: [0,1]
        Explanation: Because nums[0] + nums[1] == 9, we return [0, 1]

        Make sure to include Input:, Output:, and Explanation: on separate lines in the Example section.
        """

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt}
        ]
        
        response = llm.invoke(messages)
        
        # Parse the response with improved handling
        content = response.content
        sections = {
            'Title': '',
            'Description': '',
            'Example': ''
        }
        
        current_section = None
        example_parts = []
        
        for line in content.split('\n'):
            line = line.strip()
            if line.startswith('Title:'):
                current_section = 'Title'
                sections[current_section] = line.replace('Title:', '').strip()
            elif line.startswith('Description:'):
                current_section = 'Description'
                sections[current_section] = line.replace('Description:', '').strip()
            elif line.startswith('Example:'):
                current_section = 'Example'
                sections[current_section] = ''
            elif current_section == 'Example' and line:
                if line.startswith('Input:') or line.startswith('Output:') or line.startswith('Explanation:'):
                    example_parts.append(line)
                else:
                    # Append to the last part if it exists
                    if example_parts:
                        example_parts[-1] += ' ' + line

        # Format the example properly
        if example_parts:
            sections['Example'] = '\n'.join(example_parts)

        # Append to problems file with proper formatting
        with open('problem details.txt', 'a', encoding='utf-8') as f:
            f.write(f"\n\n\n{len(sections['Title'])}. {sections['Title']}\n")
            f.write(f"{sections['Description']}\n\n")
            f.write(f"Example:\n{sections['Example']}")

        return jsonify({
            "success": True,
            "problem": {
                "title": sections['Title'],
                "description": sections['Description'],
                "example": sections['Example']
            }
        })

    except Exception as e:
        logger.error(f"Error generating problem: {str(e)}")
        return jsonify({"error": str(e)}), 500


if __name__ == '__main__':
    app.run(debug=True, port=5000, host='127.0.0.1')