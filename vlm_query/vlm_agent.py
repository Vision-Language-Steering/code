"""
Adapted from ReKep/constraint_generation.py. [url: https://github.com/huangwl18/ReKep/blob/main/constraint_generation.py]
"""


import base64
from openai import OpenAI
import os
import cv2
import json
import parse
import numpy as np
import time
import re
from datetime import datetime

from utils.logging_utils import SteerLogger
log = SteerLogger("VLMAgent")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')


class VLMAgent:
    def __init__(self, config, base_dir=None, env_type=None):
        self.config = config
        self.env_type = env_type

        # Support Azure OpenAI or standard OpenAI
        api_key = config.get('api_key') or os.environ.get('OPENAI_API_KEY') or os.environ.get('OPENAI_API_KEY')
        base_url = config.get('base_url') or os.environ.get('AZURE_OPENAI_BASE_URL')  # For Azure OpenAI

        if base_url:
            self.client = OpenAI(api_key=api_key, base_url=base_url)
            log.info(f"Using custom OpenAI endpoint: {base_url}")
        else:
            self.client = OpenAI(api_key=api_key)
            log.info("Using standard OpenAI endpoint")

        self.base_dir = base_dir
        # self.task_dir = os.path.join(self.base_dir, "vlm_agent")
        # Don't create task_dir here - it will be created when needed
        # This allows us to set task_dir per episode without creating empty folders
        
        # Load prompt templates with base + environment-specific logic
        template_dir = self.config["query_template_dir"]
        base_template_path = os.path.join(template_dir, 'guidance_template.txt')
        
        # Determine environment-specific template
        if self.env_type:
            env_template_file = f"guidance_template_{self.env_type}.txt"
        else:
            env_template_file = self.config.get("prompt_template", "guidance_template_libero.txt")
        
        env_template_path = os.path.join(template_dir, env_template_file)
        
        # Load base template
        if os.path.exists(base_template_path):
            with open(base_template_path, 'r') as f:
                base_template = f.read()
        else:
            log.warning(f"Base template not found at {base_template_path}")
            base_template = "{task_specific_patterns}"
            
        # Load environment-specific notes/patterns
        if os.path.exists(env_template_path):
            with open(env_template_path, 'r') as f:
                env_notes = f.read()
        else:
            log.error(f"Environment template not found at {env_template_path}")
            env_notes = "(No specific patterns for this environment)"
            
        # Insert environment notes into the base template's placeholder
        # We use replace instead of format to avoid issues with code-style braces in the template
        self.prompt_template = base_template.replace("{task_specific_patterns}", env_notes)
        log.info(f"Initialized combined prompt template using {env_template_file} as task-specific notes")

        with open(os.path.join(template_dir, 'stage_template.txt'), 'r') as f:
            self.stage_template = f.read()

        # Load segmentation planning template for agentic object part selection
        seg_template_path = os.path.join(self.config["query_template_dir"], 'segmentation_planning_template.txt')
        if os.path.exists(seg_template_path):
            with open(seg_template_path, 'r') as f:
                self.segmentation_template = f.read()
        else:
            self.segmentation_template = None


    def _build_prompt(self, image_path, instruction, template=None, save_dir=None, **kwargs):
        """
        Build prompt for VLM API call.
        
        Args:
            image_path: path to the image
            instruction: task instruction
            template: prompt template to use (default: self.prompt_template)
            save_dir: directory to save prompt.txt (optional)
            **kwargs: additional format arguments for the template
        """
        img_base64 = encode_image(image_path)
        
        # Use provided template or default
        if template is None:
            template = self.prompt_template
        
        # Manually replace placeholders to avoid issues with curly braces in code examples
        prompt_text = template
        replacements = {
            "{instruction}": str(instruction),
            "{key_points_objects_map}": str(kwargs.get('key_points_objects_map', '')),
            "{init_keypoint_positions}": str(kwargs.get('init_keypoint_positions', '')),
            "{num_keypoints}": str(kwargs.get('num_keypoints', '')),
        }
        
        for placeholder, value in replacements.items():
            prompt_text = prompt_text.replace(placeholder, value)
        
        # save prompt if save_dir is provided
        if save_dir is not None:
            with open(os.path.join(save_dir, 'prompt.txt'), 'w', encoding='utf-8') as f:
                f.write(prompt_text)
        
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt_text
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/png;base64,{img_base64}"
                        }
                    },
                ]
            }
        ]
        return messages

    def _parse_and_save_guidance_functions(self, output, save_dir):
        # parse into function blocks
        lines = output.split("\n")
        functions = dict()
        for i, line in enumerate(lines):
            if line.startswith("def "):
                start = i
                name = line.split("(")[0].split("def ")[1]
            if line.startswith("    return "):
                end = i
                functions[name] = lines[start:end+1]
        # organize them based on hierarchy in function names
        groupings = dict()
        for name in functions:
            parts = name.split("_")[:-1]  # last one is the guidance function idx
            key = "_".join(parts)
            if key not in groupings:
                groupings[key] = []
            groupings[key].append(name)
        # save them into files
        for key in groupings:
            with open(os.path.join(save_dir, f"{key}_guidance.txt"), "w") as f:
                for name in groupings[key]:
                    f.write("\n".join(functions[name]) + "\n\n")
        log.info(f"Guidance Functions saved to {save_dir}")
    
    def _parse_other_metadata(self, output):
        data_dict = dict()
        
        # Find num_stages
        num_stages_template = "num_stages = {num_stages}"
        for line in output.split("\n"):
            num_stages = parse.parse(num_stages_template, line)
            if num_stages is not None:
                break
        if num_stages is None:
            raise ValueError("num_stages not found in output")
        # Remove comments from num_stages value (e.g., "1  # comment" -> "1")
        num_stages_str = num_stages['num_stages'].split('#')[0].strip()
        data_dict['num_stages'] = int(num_stages_str)
        
        # Find stage_names (optional, with fallback)
        stage_names_template = "stage_names = {stage_names}"
        stage_names = None
        for line in output.split("\n"):
            result = parse.parse(stage_names_template, line)
            if result is not None:
                stage_names = result
                break
        
        if stage_names is not None:
            # Parse list format: ["name1", "name2", "name3"]
            names_str = stage_names['stage_names'].strip()
            # Simple parsing: extract strings between quotes
            import re
            matches = re.findall(r'["\']([^"\']+)["\']', names_str)
            if matches:
                data_dict['stage_names'] = matches
                log.info(f"Parsed stage names: {matches}")
            else:
                # Fallback: generate default names
                data_dict['stage_names'] = [f"Stage {i+1}" for i in range(data_dict['num_stages'])]
                log.warning(f"Could not parse stage_names, using default")
        else:
            # Fallback: generate default names
            data_dict['stage_names'] = [f"Stage {i+1}" for i in range(data_dict['num_stages'])]
            log.info(f"stage_names not found in output, using default")

        return data_dict

    def _save_metadata(self, metadata):
        for k, v in metadata.items():
            if isinstance(v, np.ndarray):
                metadata[k] = v.tolist()
        with open(os.path.join(self.task_dir, 'metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Metadata saved to {os.path.join(self.task_dir, 'metadata.json')}")

    def generate_guidance(self, img, instruction, metadata: dict):
        """
        Args:
            img (np.ndarray): image of the scene (H, W, 3) uint8
            instruction (str): instruction for the query
            metadata (dict): metadata for the query
        Returns:
            save_dir (str): directory where the constraints
        """
        # Create task_dir if it doesn't exist (allows per-episode directories)
        os.makedirs(self.task_dir, exist_ok=True)
        
        # save query image
        image_path = os.path.join(self.task_dir, 'query_img.png')
        cv2.imwrite(image_path, img[..., ::-1])
        # build prompt
        messages = self._build_prompt(image_path, instruction, save_dir=self.task_dir, **metadata)
        # stream back the response
        stream = self.client.chat.completions.create(model=self.config['model'],
                                                        messages=messages,
                                                        temperature=self.config['temperature'],
                                                        max_completion_tokens=self.config['max_completion_tokens'],
                                                        stream=True)
        output = ""
        start = time.time()
        for chunk in stream:
            print(f'[{time.time()-start:.2f}s] Querying OpenAI API...', end='\r')
            if chunk.choices and len(chunk.choices) > 0:
                if chunk.choices[0].delta.content is not None:
                    output += chunk.choices[0].delta.content
        print(f'[{time.time()-start:.2f}s] Querying OpenAI API...Done')
        # save raw output
        with open(os.path.join(self.task_dir, 'output_raw.txt'), 'w') as f:
            f.write(output)
        # parse and save constraints
        self._parse_and_save_guidance_functions(output, self.task_dir)
        # save metadata
        metadata.update(self._parse_other_metadata(output))
        self._save_metadata(metadata)
        log.info(f"Metadata saved to {os.path.join(self.task_dir, 'metadata.json')}")
        return self.task_dir
    
    def _extract_stage_descriptions_from_output(self, output_raw_path):
        """
        Extract stage descriptions from the output_raw.txt file.
        
        Args:
            output_raw_path: path to output_raw.txt
            
        Returns:
            str: formatted stage descriptions with clear numbering
        """
        if not os.path.exists(output_raw_path):
            return ""
        
        with open(output_raw_path, 'r') as f:
            content = f.read()
        
        # Extract stage descriptions from comments
        stage_descriptions = []
        lines = content.split('\n')
        in_stages_section = False
        
        for line in lines:
            # Look for the Stages: section in comments
            if '# Stages:' in line or '# Stage breakdown:' in line or '# Task breakdown:' in line:
                in_stages_section = True
                continue
            
            # Extract stage lines that start with # followed by a number
            if in_stages_section and line.strip().startswith('#'):
                # Check for two possible formats:
                # 1. Old format: "# 1) Move to grasp..." 
                # 2. New format: "# Stage 1: Move to grasp..."
                old_format_match = re.match(r'#\s*(\d+)\)', line)
                new_format_match = re.match(r'#\s*Stage\s+(\d+):', line, re.IGNORECASE)
                
                if old_format_match or new_format_match:
                    # Clean up the line
                    stage_desc = line.strip().lstrip('#').strip()
                    stage_descriptions.append(stage_desc)
                # Stop if we reach an empty comment or code
                elif line.strip() == '#' or not line.strip().startswith('#'):
                    break
            elif in_stages_section and not line.strip().startswith('#'):
                # Exit stages section when we hit non-comment lines
                break
        
        # Format the descriptions with clear stage numbers
        if stage_descriptions:
            formatted_descriptions = []
            for i, desc in enumerate(stage_descriptions, start=1):
                # Remove existing numbering if present
                # Handle formats: "1) ...", "1. ...", "Stage 1: ..."
                desc_cleaned = re.sub(r'^\d+[\)\.]\s*', '', desc)
                desc_cleaned = re.sub(r'^Stage\s+\d+[:\s]*', '', desc_cleaned, flags=re.IGNORECASE)
                desc_cleaned = desc_cleaned.strip()
                # Add standardized numbering
                formatted_descriptions.append(f"Stage {i}: {desc_cleaned}")
            return '\n'.join(formatted_descriptions)
        else:
            # Fallback: try to extract from function comments
            log.warning("Could not extract stage descriptions from output_raw.txt")
            return ""
    
    def plan_segmentation(self, img, instruction):
        """
        Use VLM to plan what object parts to segment for the given task.
        This enables agentic, task-aware segmentation instead of hardcoded object lists.

        Args:
            img (np.ndarray): RGB image of the scene (H, W, 3) uint8
            instruction (str): task instruction (e.g., "close the drawer")

        Returns:
            List[str]: object parts to segment (e.g., ["gripper", "drawer front edge", "drawer handle"])
        """
        if self.segmentation_template is None:
            log.warning("Segmentation planning template not found, using default objects")
            return None

        # Save temp image
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as f:
            image_path = f.name

        # Ensure img is uint8
        if img.dtype != np.uint8:
            img = img.astype(np.uint8)
        cv2.imwrite(image_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))

        # Build prompt
        img_base64 = encode_image(image_path)
        prompt_text = self.segmentation_template.format(instruction=instruction)

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt_text},
                    {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{img_base64}"}}
                ]
            }
        ]

        try:
            response = self.client.chat.completions.create(
                model=self.config['model'],
                messages=messages,
                temperature=0.3,  # Lower temperature for consistent outputs
                max_completion_tokens=200
            )
            output = response.choices[0].message.content.strip()
            log.info(f"[Segmentation Planning] VLM output: {output}")

            # Parse JSON output
            # Find JSON block in response
            json_match = re.search(r'\{[^}]+\}', output, re.DOTALL)
            if json_match:
                json_str = json_match.group()
                data = json.loads(json_str)
                object_parts = data.get('object_parts', [])
                log.info(f"[Segmentation Planning] Planned object parts: {object_parts}")
                return object_parts
            else:
                log.warning(f"Could not parse JSON from output: {output}")
                return None

        except Exception as e:
            log.warning(f"Error in segmentation planning: {e}")
            return None
        finally:
            if os.path.exists(image_path):
                os.remove(image_path)


if __name__ == "__main__":
    # Configuration
    config = {
        'model': 'gpt-5.1', 
        'temperature': 0.7,
        'max_completion_tokens': 2000
    }
    
    # Test 1: Guidance Generator
    print("=" * 50)
    print("Testing Guidance Generator")
    print("=" * 50)
    
    # You need to provide these
    test_image_path = "/tmp/test_keypoints.png"  # Update with actual test image path
    if os.path.exists(test_image_path):
        img = cv2.imread(test_image_path)[..., ::-1]  # BGR to RGB
        instruction = "put red block on the green tile".strip()
        
        # Create some dummy metadata
        metadata = {
            'timestamp': datetime.now().isoformat(),
            'instruction': instruction
        }
        
        agent = VLMAgent(config)
        try:
            task_dir = agent.generate_guidance(img, instruction, metadata)
            print(f"Guidance generation completed. Results saved to: {task_dir}")
        except Exception as e:
            print(f"Error in guidance generation: {e}")
    else:
        print(f"Image not found: {test_image_path}")
    
    print("\n")
    
    # Test 2: Stage Recognizer (Gemini)
    # This is now handled by GeminiStageRecognizer in core/gemini_grounder.py
