z_image_enhancer_en = '''import requests
import json
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)

class ZImagePromptEnhancer:
    """
    ComfyUI Custom Node for Z-Image Prompt Enhancement with local Ollama LLM.
    Transforms amateur prompts into professional Z-Image-optimized prompts.
    """
    
    # Z-Image Best Practices System Prompt
    Z_IMAGE_SYSTEM_PROMPT = """You are an expert in prompt engineering for Z-Image-Turbo, an advanced AI image generation system.

Your task is to transform amateur prompts into professional, detailed prompts optimized for Z-Image.

Z-Image Best Practices:
- Long, descriptive prompts (300-500 characters optimal)
- Specific details: materials, textures, lighting, mood, camera angle
- Photography and cinematography terminology
- Precise composition and color palettes
- Quality indicators at the end
- Structured: Subject + Action + Environment + Lighting + Style + Quality

Return ONLY the enhanced prompt, no explanations or markdown."""
    
    # Style Templates for Z-Image
    STYLE_TEMPLATES = {
        "photorealistic": "Photorealistic, documentary-style photography with natural lighting, authentic details, professional camera work, sharp focus",
        "cinematic": "Cinematic, film quality with dramatic lighting, color grading and professional cinematography, 35mm film aesthetic",
        "artistic": "Artistic, painterly style with expressive brushstrokes, creative interpretation and emotional depth",
        "illustration": "Illustration style, animated or hand-drawn with clean lines, vibrant colors and distinctive shapes",
        "3d": "3D rendered, high-poly 3D modeling style with perfect lighting, materials and textures, ray-traced quality",
        "fantasy": "Fantasy style with mythological elements, magic, dramatic atmosphere and detailed world-building",
        "steampunk": "Steampunk aesthetic with Victorian design, brass, gears, steam and warm lighting, mechanical intricate details"
    }
    
    # Negative Prompts for each Style
    NEGATIVE_TEMPLATES = {
        "photorealistic": "blurry, low quality, distorted, ugly, bad anatomy, watermark, text, cartoon, painting, illustration, artificial, fake, unnatural, CGI",
        "cinematic": "blurry, low quality, distorted, flat lighting, overexposed, underexposed, artificial colors, anime, video game",
        "artistic": "blurry, low quality, photorealistic, 3D render, digital artifacts, computer generated look",
        "illustration": "blurry, low quality, photorealistic, overly realistic, CGI, 3D render, photograph",
        "3d": "blurry, low quality, 2D, flat, painting, cartoon, unrendered, wireframe, sketch",
        "fantasy": "blurry, low quality, realistic, modern, boring, photorealistic without magic, mundane",
        "steampunk": "blurry, low quality, modern, futuristic, cyberpunk, clean, sterile, realistic without steampunk elements"
    }
    
    # Enhancement Level Instructions
    ENHANCEMENT_INSTRUCTIONS = {
        "light": "Slightly enhance this prompt with more detail. Keep the overall tone similar to the original.",
        "moderate": "Enhance this prompt with vivid details, color descriptions and mood. This is the standard level.",
        "heavy": "Create an extremely detailed, cinematic prompt with specific camera angles, lighting, materials, atmosphere and textures. Maximally elaborate."
    }
    
    def __init__(self):
        pass
    
    @classmethod
    def INPUT_TYPES(cls):
        """Define input parameters for the node"""
        return {
            "required": {
                "prompt": ("STRING", {
                    "multiline": True,
                    "default": "a woman in the forest"
                }),
                "ollama_url": ("STRING", {
                    "default": "http://localhost:11434"
                }),
                "model": (["mistral", "neural-chat", "qwen", "phi3", "llama2", "openchat"],),
                "style": (list(cls.STYLE_TEMPLATES.keys()),),
                "enhancement_level": (["light", "moderate", "heavy"],),
            },
            "optional": {
                "quality_tags": ("STRING", {
                    "default": "highly detailed, sharp focus, 8K resolution"
                }),
                "aspect_ratio": (["1:1", "16:9", "9:16", "4:3", "3:4"],),
                "temperature": ("FLOAT", {
                    "default": 0.7,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
                "top_p": ("FLOAT", {
                    "default": 0.9,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.1
                }),
            }
        }
    
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("enhanced_prompt", "negative_prompt")
    FUNCTION = "enhance"
    CATEGORY = "Z-Image"
    
    def enhance(self, prompt: str, ollama_url: str, model: str, style: str, 
                enhancement_level: str, quality_tags: str = "", aspect_ratio: str = "1:1",
                temperature: float = 0.7, top_p: float = 0.9) -> Tuple[str, str]:
        """
        Enhance user prompt with Ollama LLM
        """
        
        try:
            # Validate inputs
            if not prompt or prompt.strip() == "":
                return ("Error: Prompt cannot be empty", "")
            
            # Build style instruction
            style_instruction = self.STYLE_TEMPLATES.get(style, self.STYLE_TEMPLATES["photorealistic"])
            
            # Build system prompt
            system_prompt = f"""{self.Z_IMAGE_SYSTEM_PROMPT}

Style: {style_instruction}
Enhancement Level: {enhancement_level}
Aspect Ratio: {aspect_ratio}
Quality Tags: {quality_tags}"""
            
            # Build user message
            enhancement_instruction = self.ENHANCEMENT_INSTRUCTIONS.get(enhancement_level, self.ENHANCEMENT_INSTRUCTIONS["moderate"])
            user_message = f"""Original prompt: "{prompt}"

{enhancement_instruction}

Create an enhanced prompt optimized for Z-Image-Turbo generation."""
            
            # Call Ollama API
            logger.info(f"Calling Ollama at {ollama_url} with model {model}")
            response = self._call_ollama(
                ollama_url=ollama_url,
                model=model,
                system_prompt=system_prompt,
                user_message=user_message,
                temperature=temperature,
                top_p=top_p
            )
            
            if not response:
                return ("Error: No response from Ollama. Check if server is running.", "")
            
            enhanced_prompt = response.strip()
            
            # Generate negative prompt
            negative_prompt = self.NEGATIVE_TEMPLATES.get(style, self.NEGATIVE_TEMPLATES["photorealistic"])
            
            logger.info(f"Enhanced prompt generated successfully")
            return (enhanced_prompt, negative_prompt)
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            logger.error(error_msg)
            return (error_msg, "")
    
    def _call_ollama(self, ollama_url: str, model: str, system_prompt: str, 
                     user_message: str, temperature: float, top_p: float) -> str:
        """Call Ollama API and get response"""
        
        try:
            # Prepare request
            payload = {
                "model": model,
                "prompt": user_message,
                "system": system_prompt,
                "stream": False,
                "temperature": temperature,
                "top_p": top_p
            }
            
            # Make request
            response = requests.post(
                f"{ollama_url}/api/generate",
                json=payload,
                timeout=300  # 5 minutes timeout
            )
            
            # Check response
            if response.status_code != 200:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
            
            # Extract text
            result = response.json()
            return result.get("response", "")
            
        except requests.exceptions.ConnectionError:
            logger.error(f"Cannot connect to Ollama at {ollama_url}")
            return None
        except requests.exceptions.Timeout:
            logger.error("Ollama request timed out")
            return None
        except Exception as e:
            logger.error(f"Ollama API call failed: {str(e)}")
            return None


class ZImageNegativePromptGenerator:
    """
    Standalone node to quickly generate negative prompts for different styles
    """
    
    NEGATIVE_TEMPLATES = ZImagePromptEnhancer.NEGATIVE_TEMPLATES
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style": (list(cls.NEGATIVE_TEMPLATES.keys()),),
                "custom_negative": ("STRING", {
                    "multiline": True,
                    "default": ""
                }),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("negative_prompt",)
    FUNCTION = "generate"
    CATEGORY = "Z-Image"
    
    def generate(self, style: str, custom_negative: str = "") -> Tuple[str]:
        base_negative = self.NEGATIVE_TEMPLATES.get(style, "")
        
        if custom_negative:
            return (f"{base_negative}, {custom_negative}",)
        
        return (base_negative,)


class ZImageStyleDescriber:
    """
    Shows style descriptions and best practices
    """
    
    STYLE_TEMPLATES = ZImagePromptEnhancer.STYLE_TEMPLATES
    
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "style": (list(cls.STYLE_TEMPLATES.keys()),),
            }
        }
    
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("style_description",)
    FUNCTION = "describe"
    CATEGORY = "Z-Image"
    
    def describe(self, style: str) -> Tuple[str]:
        description = self.STYLE_TEMPLATES.get(style, "Unknown style")
        return (description,)


# Register nodes
NODE_CLASS_MAPPINGS = {
    "Z-Image Prompt Enhancer": ZImagePromptEnhancer,
    "Z-Image Negative Prompt Generator": ZImageNegativePromptGenerator,
    "Z-Image Style Describer": ZImageStyleDescriber,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Z-Image Prompt Enhancer": "🎨 Z-Image Prompt Enhancer (Ollama)",
    "Z-Image Negative Prompt Generator": "🚫 Z-Image Negative Prompt Generator",
    "Z-Image Style Describer": "📖 Z-Image Style Describer",
}
'''

init_py_en = '''from .z_image_enhancer import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
'''

requirements_txt = '''requests>=2.28.0
'''

readme_en = '''# ComfyUI Z-Image Prompt Enhancer

A specialized custom node for ComfyUI that transforms prompts into professional Z-Image-optimized prompts using a local Ollama LLM.

## Features

- ✅ Local Ollama LLM integration (no external API)
- ✅ 7 predefined styles (photorealistic, cinematic, artistic, etc.)
- ✅ 3 enhancement levels (light, moderate, heavy)
- ✅ Automatic negative prompt generation
- ✅ Quality tags and aspect ratio support
- ✅ Temperature & Top-P control for fine-tuning

## Installation

### Via ComfyUI Manager (Recommended)
1. Open ComfyUI Manager
2. Search for "Z-Image Prompt Enhancer"
3. Click Install

### Manual Installation
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/yourusername/ComfyUI-ZImageEnhancer.git
pip install -r ComfyUI-ZImageEnhancer/requirements.txt
```

## Requirements

- Ollama installed and running (`ollama serve`)
- A model downloaded: `ollama pull mistral` or `ollama pull neural-chat`

## Usage

1. Add "🎨 Z-Image Prompt Enhancer (Ollama)" node to your workflow
2. Enter your prompt (even if it's simple/amateur)
3. Select your desired style, enhancement level, and model
4. Click "Execute"
5. The enhanced prompt and negative prompt will appear

## Recommended Models

- **Mistral 7B**: Fast and good quality (8-16GB RAM)
- **Neural-Chat 7B**: Specialized for text generation (8-16GB RAM)
- **Qwen 2.5 7B**: Very competent, slightly slower (8-16GB RAM)
- **Phi 3 3.8B**: Lightweight, less powerful (4-8GB RAM)

## Available Styles

1. **photorealistic** - Professional photography style
2. **cinematic** - Film-quality with dramatic lighting
3. **artistic** - Painterly style with creative interpretation
4. **illustration** - Animated or hand-drawn aesthetic
5. **3d** - 3D rendered with perfect materials
6. **fantasy** - Mythological elements and magic
7. **steampunk** - Victorian-era mechanical aesthetic

## Enhancement Levels

- **Light** (~5-10s): Subtle improvements, stays close to original
- **Moderate** (~10-20s): Good balance, adds vivid details (recommended)
- **Heavy** (~20-40s): Maximum detail, very elaborate descriptions

*Timings depend on model and hardware*

## Example Workflow

```
[Text Input: "a woman in forest"]
        ↓
[🎨 Z-Image Prompt Enhancer]
        ↓
[enhanced_prompt] → [Z-Image Model Input]
[negative_prompt] → [Negative Prompt Input]
        ↓
[Beautiful Image Output]
```

## Node Inputs

### Required
- **prompt** (STRING): Your input prompt to enhance
- **ollama_url** (STRING): Ollama server URL (default: http://localhost:11434)
- **model** (CHOICE): LLM model to use
- **style** (CHOICE): Z-Image style template
- **enhancement_level** (CHOICE): light/moderate/heavy

### Optional
- **quality_tags** (STRING): Quality indicators (default: "highly detailed, sharp focus, 8K resolution")
- **aspect_ratio** (CHOICE): 1:1, 16:9, 9:16, 4:3, 3:4
- **temperature** (FLOAT): 0.0-1.0, controls creativity (default: 0.7)
- **top_p** (FLOAT): 0.0-1.0, controls diversity (default: 0.9)

## Node Outputs

- **enhanced_prompt** (STRING): Your improved prompt optimized for Z-Image
- **negative_prompt** (STRING): Auto-generated negative prompt for the selected style

## Setup Guide

### 1. Install Ollama
Download from https://ollama.com

### 2. Download a Model
```bash
# Recommended:
ollama pull neural-chat

# Or try others:
ollama pull mistral
ollama pull qwen
```

### 3. Start Ollama Server
```bash
ollama serve
```

### 4. Install Custom Node
Follow the installation instructions above

### 5. Restart ComfyUI
The node will appear in the "Z-Image" category

### 6. Create Your Workflow
- Add the enhancer node
- Connect to Z-Image model
- Generate beautiful images!

## Performance Tips

- **Light enhancement** works best for quick iterations
- **Moderate enhancement** is recommended for most use cases
- **Heavy enhancement** is perfect for final, detailed generations
- Smaller models (Phi 3) are faster but less detailed
- Larger models (Qwen, Mistral Nemo) are slower but better quality

## Troubleshooting

### "Cannot connect to Ollama"
```bash
# Make sure Ollama is running:
ollama serve

# Check if you can access it:
curl http://localhost:11434/api/tags
```

### "Model not found"
```bash
# List available models:
ollama list

# Download a model:
ollama pull neural-chat
```

### Node doesn't appear after installation
1. Make sure `requirements.txt` is installed: `pip install -r requirements.txt`
2. Restart ComfyUI completely
3. Clear ComfyUI cache: delete `web/scripts/__pycache__`

## Bonus Nodes

This package includes two additional helper nodes:

### 🚫 Z-Image Negative Prompt Generator
Quickly generate negative prompts for any style, with optional custom additions.

### 📖 Z-Image Style Describer
Display the style description to understand what each style will produce.

## Z-Image Best Practices

For best results with Z-Image:
- Use longer prompts (300-500 characters)
- Be specific about materials, textures, and lighting
- Include camera/photography terms
- Describe mood and atmosphere
- End with quality indicators
- Use the negative prompt to exclude unwanted elements

## License

MIT License

## Contributing

Contributions are welcome! Feel free to submit pull requests or issues.

## Support

For issues or questions, please open an issue on GitHub or contact the developer.

---

**Happy Prompting! 🎨**
'''

print("=== z_image_enhancer.py ===")
print(z_image_enhancer_en[:500])
print(f"\n... (Total length: {len(z_image_enhancer_en)} characters)")
print("\n=== __init__.py ===")
print(init_py_en)
print("\n=== requirements.txt ===")
print(requirements_txt)
print("\n=== README.md (preview) ===")
print(readme_en[:800])
print(f"\n... (Total length: {len(readme_en)} characters)")
