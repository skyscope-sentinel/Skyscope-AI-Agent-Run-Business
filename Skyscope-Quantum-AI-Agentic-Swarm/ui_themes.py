import streamlit as st
import base64
import json
import time
import random
import math
import os
import re
from enum import Enum
from typing import Dict, List, Any, Optional, Union, Tuple
from dataclasses import dataclass, field
from pathlib import Path
import colorsys

# Define constants
STATIC_DIR = Path("static")
FONTS_DIR = STATIC_DIR / "fonts"
THEMES_DIR = STATIC_DIR / "themes"
ANIMATIONS_DIR = STATIC_DIR / "animations"

# Ensure directories exist
for directory in [STATIC_DIR, FONTS_DIR, THEMES_DIR, ANIMATIONS_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

class ThemeType(Enum):
    """Enumeration of available UI themes."""
    DARK_GLASS = "dark_glass"
    LIGHT_GLASS = "light_glass"
    DARK_SOLID = "dark_solid"
    LIGHT_SOLID = "light_solid"
    CYBERPUNK = "cyberpunk"
    MINIMAL = "minimal"
    NEON = "neon"
    TERMINAL = "terminal"
    HOLOGRAPHIC = "holographic"
    QUANTUM = "quantum"
    CUSTOM = "custom"

class FontStyle(Enum):
    """Enumeration of available font styles."""
    OCR = "ocr"
    FUTURISTIC = "futuristic"
    TERMINAL = "terminal"
    CLEAN = "clean"
    QUANTUM = "quantum"
    MINIMAL = "minimal"
    CUSTOM = "custom"

class AnimationType(Enum):
    """Enumeration of available animation types."""
    FADE = "fade"
    SLIDE = "slide"
    PULSE = "pulse"
    GLOW = "glow"
    WAVE = "wave"
    FLICKER = "flicker"
    GRADIENT_SHIFT = "gradient_shift"
    TYPING = "typing"
    SCAN_LINE = "scan_line"
    HOLOGRAM = "hologram"
    GLITCH = "glitch"
    NONE = "none"

@dataclass
class ColorScheme:
    """Represents a color scheme for a theme."""
    primary: str
    secondary: str
    background: str
    card_bg: str
    text: str
    accent: str
    success: str
    warning: str
    error: str
    gradient_start: str
    gradient_end: str
    
    @classmethod
    def dark_glass(cls) -> 'ColorScheme':
        """Default dark glass theme colors."""
        return cls(
            primary="#00FFFF",
            secondary="#FF00FF",
            background="#0A0A1F",
            card_bg="rgba(20, 20, 40, 0.7)",
            text="#FFFFFF",
            accent="#00CCFF",
            success="#00FF9F",
            warning="#FFCC00",
            error="#FF3366",
            gradient_start="#00FFFF",
            gradient_end="#FF00FF"
        )
    
    @classmethod
    def light_glass(cls) -> 'ColorScheme':
        """Light glass theme colors."""
        return cls(
            primary="#0088FF",
            secondary="#FF0088",
            background="#F0F0FF",
            card_bg="rgba(240, 240, 255, 0.7)",
            text="#0A0A1F",
            accent="#0066CC",
            success="#00CC7A",
            warning="#FF9900",
            error="#FF0044",
            gradient_start="#0088FF",
            gradient_end="#FF0088"
        )
    
    @classmethod
    def cyberpunk(cls) -> 'ColorScheme':
        """Cyberpunk theme colors."""
        return cls(
            primary="#00FFBB",
            secondary="#FF00BB",
            background="#0A0A1F",
            card_bg="rgba(10, 10, 31, 0.8)",
            text="#EEFF00",
            accent="#FF00BB",
            success="#00FFBB",
            warning="#FFBB00",
            error="#FF0055",
            gradient_start="#00FFBB",
            gradient_end="#FF00BB"
        )
    
    @classmethod
    def terminal(cls) -> 'ColorScheme':
        """Terminal theme colors."""
        return cls(
            primary="#00FF00",
            secondary="#006600",
            background="#000000",
            card_bg="rgba(0, 20, 0, 0.7)",
            text="#00FF00",
            accent="#00DD00",
            success="#00FF00",
            warning="#FFFF00",
            error="#FF0000",
            gradient_start="#003300",
            gradient_end="#00FF00"
        )
    
    @classmethod
    def holographic(cls) -> 'ColorScheme':
        """Holographic theme colors."""
        return cls(
            primary="#88DDFF",
            secondary="#FF88DD",
            background="#0A0A1F",
            card_bg="rgba(20, 20, 40, 0.3)",
            text="#FFFFFF",
            accent="#88FFDD",
            success="#88FFAA",
            warning="#FFDD88",
            error="#FF88AA",
            gradient_start="#88DDFF",
            gradient_end="#FF88DD"
        )
    
    @classmethod
    def quantum(cls) -> 'ColorScheme':
        """Quantum theme colors."""
        return cls(
            primary="#00DDFF",
            secondary="#AA00FF",
            background="#0A0A1F",
            card_bg="rgba(10, 10, 31, 0.6)",
            text="#FFFFFF",
            accent="#00FFDD",
            success="#00FFAA",
            warning="#FFAA00",
            error="#FF00AA",
            gradient_start="#00DDFF",
            gradient_end="#AA00FF"
        )
    
    def generate_variants(self) -> Dict[str, str]:
        """Generate color variants for the theme."""
        variants = {}
        
        # Generate lighter and darker variants
        for name, color in self.__dict__.items():
            if name.startswith("_") or not isinstance(color, str) or not color.startswith("#"):
                continue
            
            # Skip rgba colors
            if "rgba" in color:
                variants[f"{name}_lighter"] = color
                variants[f"{name}_darker"] = color
                continue
            
            # Convert hex to RGB
            r = int(color[1:3], 16) / 255.0
            g = int(color[3:5], 16) / 255.0
            b = int(color[5:7], 16) / 255.0
            
            # Convert RGB to HSL
            h, l, s = colorsys.rgb_to_hls(r, g, b)
            
            # Generate lighter variant (increase lightness)
            l_lighter = min(1.0, l * 1.3)
            r_lighter, g_lighter, b_lighter = colorsys.hls_to_rgb(h, l_lighter, s)
            variants[f"{name}_lighter"] = "#{:02x}{:02x}{:02x}".format(
                int(r_lighter * 255), int(g_lighter * 255), int(b_lighter * 255)
            )
            
            # Generate darker variant (decrease lightness)
            l_darker = max(0.0, l * 0.7)
            r_darker, g_darker, b_darker = colorsys.hls_to_rgb(h, l_darker, s)
            variants[f"{name}_darker"] = "#{:02x}{:02x}{:02x}".format(
                int(r_darker * 255), int(g_darker * 255), int(b_darker * 255)
            )
        
        return variants

@dataclass
class FontSet:
    """Represents a set of fonts for a theme."""
    primary: str
    secondary: str
    monospace: str
    heading: str
    display: str
    
    @classmethod
    def ocr(cls) -> 'FontSet':
        """OCR-style font set."""
        return cls(
            primary="'OCR A Extended', 'Share Tech Mono', monospace",
            secondary="'Share Tech Mono', 'OCR A Extended', monospace",
            monospace="'Share Tech Mono', 'Courier New', monospace",
            heading="'Orbitron', 'OCR A Extended', sans-serif",
            display="'Orbitron', 'Share Tech Mono', sans-serif"
        )
    
    @classmethod
    def futuristic(cls) -> 'FontSet':
        """Futuristic font set."""
        return cls(
            primary="'Orbitron', 'Rajdhani', sans-serif",
            secondary="'Rajdhani', 'Orbitron', sans-serif",
            monospace="'Share Tech Mono', 'Courier New', monospace",
            heading="'Orbitron', 'Rajdhani', sans-serif",
            display="'Orbitron', 'Rajdhani', sans-serif"
        )
    
    @classmethod
    def terminal(cls) -> 'FontSet':
        """Terminal font set."""
        return cls(
            primary="'VT323', 'Share Tech Mono', monospace",
            secondary="'Share Tech Mono', 'VT323', monospace",
            monospace="'Share Tech Mono', 'Courier New', monospace",
            heading="'VT323', 'Share Tech Mono', monospace",
            display="'VT323', 'Share Tech Mono', monospace"
        )
    
    @classmethod
    def clean(cls) -> 'FontSet':
        """Clean modern font set."""
        return cls(
            primary="'Roboto', 'Segoe UI', sans-serif",
            secondary="'Roboto Condensed', 'Segoe UI', sans-serif",
            monospace="'Roboto Mono', 'Courier New', monospace",
            heading="'Rajdhani', 'Roboto', sans-serif",
            display="'Rajdhani', 'Roboto', sans-serif"
        )

@dataclass
class AnimationSet:
    """Represents a set of animations for a theme."""
    loading: AnimationType
    transition: AnimationType
    highlight: AnimationType
    background: AnimationType
    button: AnimationType
    card: AnimationType
    text: AnimationType
    duration_scale: float = 1.0
    
    @classmethod
    def default(cls) -> 'AnimationSet':
        """Default animation set."""
        return cls(
            loading=AnimationType.PULSE,
            transition=AnimationType.FADE,
            highlight=AnimationType.GLOW,
            background=AnimationType.GRADIENT_SHIFT,
            button=AnimationType.PULSE,
            card=AnimationType.FADE,
            text=AnimationType.NONE,
            duration_scale=1.0
        )
    
    @classmethod
    def minimal(cls) -> 'AnimationSet':
        """Minimal animation set."""
        return cls(
            loading=AnimationType.PULSE,
            transition=AnimationType.FADE,
            highlight=AnimationType.NONE,
            background=AnimationType.NONE,
            button=AnimationType.NONE,
            card=AnimationType.NONE,
            text=AnimationType.NONE,
            duration_scale=0.5
        )
    
    @classmethod
    def cyberpunk(cls) -> 'AnimationSet':
        """Cyberpunk animation set."""
        return cls(
            loading=AnimationType.GLITCH,
            transition=AnimationType.SLIDE,
            highlight=AnimationType.FLICKER,
            background=AnimationType.SCAN_LINE,
            button=AnimationType.GLOW,
            card=AnimationType.GLITCH,
            text=AnimationType.TYPING,
            duration_scale=1.2
        )
    
    @classmethod
    def holographic(cls) -> 'AnimationSet':
        """Holographic animation set."""
        return cls(
            loading=AnimationType.HOLOGRAM,
            transition=AnimationType.FADE,
            highlight=AnimationType.GLOW,
            background=AnimationType.WAVE,
            button=AnimationType.PULSE,
            card=AnimationType.HOLOGRAM,
            text=AnimationType.FADE,
            duration_scale=1.5
        )

@dataclass
class UITheme:
    """Represents a complete UI theme."""
    name: str
    type: ThemeType
    colors: ColorScheme
    fonts: FontSet
    animations: AnimationSet
    border_radius: str = "10px"
    shadow_intensity: float = 1.0
    glass_opacity: float = 0.7
    glass_blur: str = "10px"
    border_width: str = "1px"
    custom_css: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert theme to dictionary for serialization."""
        return {
            "name": self.name,
            "type": self.type.value,
            "colors": {k: v for k, v in self.colors.__dict__.items()},
            "fonts": {k: v for k, v in self.fonts.__dict__.items()},
            "animations": {
                "loading": self.animations.loading.value,
                "transition": self.animations.transition.value,
                "highlight": self.animations.highlight.value,
                "background": self.animations.background.value,
                "button": self.animations.button.value,
                "card": self.animations.card.value,
                "text": self.animations.text.value,
                "duration_scale": self.animations.duration_scale
            },
            "border_radius": self.border_radius,
            "shadow_intensity": self.shadow_intensity,
            "glass_opacity": self.glass_opacity,
            "glass_blur": self.glass_blur,
            "border_width": self.border_width,
            "custom_css": self.custom_css
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UITheme':
        """Create theme from dictionary."""
        colors = ColorScheme(
            primary=data["colors"]["primary"],
            secondary=data["colors"]["secondary"],
            background=data["colors"]["background"],
            card_bg=data["colors"]["card_bg"],
            text=data["colors"]["text"],
            accent=data["colors"]["accent"],
            success=data["colors"]["success"],
            warning=data["colors"]["warning"],
            error=data["colors"]["error"],
            gradient_start=data["colors"]["gradient_start"],
            gradient_end=data["colors"]["gradient_end"]
        )
        
        fonts = FontSet(
            primary=data["fonts"]["primary"],
            secondary=data["fonts"]["secondary"],
            monospace=data["fonts"]["monospace"],
            heading=data["fonts"]["heading"],
            display=data["fonts"]["display"]
        )
        
        animations = AnimationSet(
            loading=AnimationType(data["animations"]["loading"]),
            transition=AnimationType(data["animations"]["transition"]),
            highlight=AnimationType(data["animations"]["highlight"]),
            background=AnimationType(data["animations"]["background"]),
            button=AnimationType(data["animations"]["button"]),
            card=AnimationType(data["animations"]["card"]),
            text=AnimationType(data["animations"]["text"]),
            duration_scale=data["animations"]["duration_scale"]
        )
        
        return cls(
            name=data["name"],
            type=ThemeType(data["type"]),
            colors=colors,
            fonts=fonts,
            animations=animations,
            border_radius=data.get("border_radius", "10px"),
            shadow_intensity=data.get("shadow_intensity", 1.0),
            glass_opacity=data.get("glass_opacity", 0.7),
            glass_blur=data.get("glass_blur", "10px"),
            border_width=data.get("border_width", "1px"),
            custom_css=data.get("custom_css", "")
        )
    
    def save(self, directory: Path = THEMES_DIR) -> Path:
        """Save theme to file."""
        directory.mkdir(parents=True, exist_ok=True)
        safe_name = self.name.lower().replace(" ", "_")
        filepath = directory / f"{safe_name}.json"
        
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)
        
        return filepath
    
    @classmethod
    def load(cls, filepath: Path) -> 'UITheme':
        """Load theme from file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        return cls.from_dict(data)

class AnimationController:
    """Controller for UI animations."""
    
    def __init__(self, theme: UITheme):
        self.theme = theme
        self.animations = {}
        self.initialize_animations()
    
    def initialize_animations(self):
        """Initialize animation definitions based on theme."""
        duration_scale = self.theme.animations.duration_scale
        
        # Loading animations
        self.animations["loading"] = {
            AnimationType.PULSE: {
                "css": """
                @keyframes pulse {
                    0% { opacity: 0.6; }
                    50% { opacity: 1; }
                    100% { opacity: 0.6; }
                }
                .loading-pulse {
                    animation: pulse 1.5s infinite;
                }
                """,
                "duration": 1.5 * duration_scale
            },
            AnimationType.SPIN: {
                "css": """
                @keyframes spin {
                    0% { transform: rotate(0deg); }
                    100% { transform: rotate(360deg); }
                }
                .loading-spin {
                    animation: spin 1s linear infinite;
                }
                """,
                "duration": 1.0 * duration_scale
            },
            AnimationType.GLITCH: {
                "css": """
                @keyframes glitch {
                    0% { transform: translate(0); }
                    20% { transform: translate(-2px, 2px); }
                    40% { transform: translate(-2px, -2px); }
                    60% { transform: translate(2px, 2px); }
                    80% { transform: translate(2px, -2px); }
                    100% { transform: translate(0); }
                }
                .loading-glitch {
                    animation: glitch 0.5s infinite;
                }
                """,
                "duration": 0.5 * duration_scale
            },
            AnimationType.HOLOGRAM: {
                "css": """
                @keyframes hologram {
                    0% { opacity: 0.5; filter: blur(0px) hue-rotate(0deg); }
                    25% { opacity: 0.7; filter: blur(1px) hue-rotate(90deg); }
                    50% { opacity: 0.9; filter: blur(0px) hue-rotate(180deg); }
                    75% { opacity: 0.7; filter: blur(1px) hue-rotate(270deg); }
                    100% { opacity: 0.5; filter: blur(0px) hue-rotate(360deg); }
                }
                .loading-hologram {
                    animation: hologram 3s infinite;
                }
                """,
                "duration": 3.0 * duration_scale
            }
        }
        
        # Transition animations
        self.animations["transition"] = {
            AnimationType.FADE: {
                "css": """
                @keyframes fade-in {
                    from { opacity: 0; }
                    to { opacity: 1; }
                }
                .transition-fade-in {
                    animation: fade-in 0.3s ease-in-out forwards;
                }
                @keyframes fade-out {
                    from { opacity: 1; }
                    to { opacity: 0; }
                }
                .transition-fade-out {
                    animation: fade-out 0.3s ease-in-out forwards;
                }
                """,
                "duration": 0.3 * duration_scale
            },
            AnimationType.SLIDE: {
                "css": """
                @keyframes slide-in {
                    from { transform: translateX(-20px); opacity: 0; }
                    to { transform: translateX(0); opacity: 1; }
                }
                .transition-slide-in {
                    animation: slide-in 0.3s ease-out forwards;
                }
                @keyframes slide-out {
                    from { transform: translateX(0); opacity: 1; }
                    to { transform: translateX(20px); opacity: 0; }
                }
                .transition-slide-out {
                    animation: slide-out 0.3s ease-in forwards;
                }
                """,
                "duration": 0.3 * duration_scale
            }
        }
        
        # Highlight animations
        self.animations["highlight"] = {
            AnimationType.GLOW: {
                "css": """
                @keyframes glow {
                    0% { box-shadow: 0 0 5px rgba(var(--primary-rgb), 0.5); }
                    50% { box-shadow: 0 0 20px rgba(var(--primary-rgb), 0.8); }
                    100% { box-shadow: 0 0 5px rgba(var(--primary-rgb), 0.5); }
                }
                .highlight-glow {
                    animation: glow 2s infinite;
                }
                """,
                "duration": 2.0 * duration_scale
            },
            AnimationType.FLICKER: {
                "css": """
                @keyframes flicker {
                    0% { opacity: 1; }
                    5% { opacity: 0.8; }
                    10% { opacity: 1; }
                    15% { opacity: 0.3; }
                    20% { opacity: 1; }
                    80% { opacity: 1; }
                    85% { opacity: 0.7; }
                    90% { opacity: 1; }
                    95% { opacity: 0.5; }
                    100% { opacity: 1; }
                }
                .highlight-flicker {
                    animation: flicker 5s infinite;
                }
                """,
                "duration": 5.0 * duration_scale
            }
        }
        
        # Background animations
        self.animations["background"] = {
            AnimationType.GRADIENT_SHIFT: {
                "css": """
                @keyframes gradient-shift {
                    0% { background-position: 0% 50%; }
                    50% { background-position: 100% 50%; }
                    100% { background-position: 0% 50%; }
                }
                .background-gradient-shift {
                    background-size: 200% 200%;
                    animation: gradient-shift 10s ease infinite;
                }
                """,
                "duration": 10.0 * duration_scale
            },
            AnimationType.SCAN_LINE: {
                "css": """
                @keyframes scan-line {
                    0% { background-position: 0 -100vh; }
                    80% { background-position: 0 100vh; }
                    80.1% { background-position: 0 -100vh; }
                    100% { background-position: 0 -100vh; }
                }
                .background-scan-line::after {
                    content: "";
                    position: absolute;
                    top: 0;
                    left: 0;
                    right: 0;
                    bottom: 0;
                    background: linear-gradient(to bottom, transparent, rgba(var(--primary-rgb), 0.2) 50%, transparent);
                    background-size: 100% 5px;
                    pointer-events: none;
                    z-index: 10;
                    animation: scan-line 8s linear infinite;
                }
                """,
                "duration": 8.0 * duration_scale
            },
            AnimationType.WAVE: {
                "css": """
                @keyframes wave {
                    0% { background-position: 0% 50%; }
                    50% { background-position: 100% 25%; }
                    100% { background-position: 0% 50%; }
                }
                .background-wave {
                    background-image: 
                        radial-gradient(circle at 10% 20%, rgba(var(--primary-rgb), 0.1) 0%, transparent 20%),
                        radial-gradient(circle at 90% 80%, rgba(var(--secondary-rgb), 0.1) 0%, transparent 20%);
                    background-size: 200% 200%;
                    background-attachment: fixed;
                    animation: wave 15s ease infinite;
                }
                """,
                "duration": 15.0 * duration_scale
            }
        }
        
        # Text animations
        self.animations["text"] = {
            AnimationType.TYPING: {
                "css": """
                @keyframes typing {
                    from { width: 0; }
                    to { width: 100%; }
                }
                .text-typing {
                    overflow: hidden;
                    white-space: nowrap;
                    border-right: 2px solid var(--primary-color);
                    animation: typing 3s steps(40, end) forwards, blink-caret 0.75s step-end infinite;
                }
                @keyframes blink-caret {
                    from, to { border-color: transparent; }
                    50% { border-color: var(--primary-color); }
                }
                """,
                "duration": 3.0 * duration_scale
            }
        }
    
    def get_animation_css(self, animation_type: AnimationType, category: str) -> str:
        """Get CSS for a specific animation."""
        if category not in self.animations:
            return ""
        
        if animation_type not in self.animations[category]:
            return ""
        
        return self.animations[category][animation_type]["css"]
    
    def get_all_animations_css(self) -> str:
        """Get CSS for all animations used in the theme."""
        css = ""
        
        # Add animations based on theme configuration
        categories = {
            "loading": self.theme.animations.loading,
            "transition": self.theme.animations.transition,
            "highlight": self.theme.animations.highlight,
            "background": self.theme.animations.background,
            "button": self.theme.animations.button,
            "card": self.theme.animations.card,
            "text": self.theme.animations.text
        }
        
        for category, animation_type in categories.items():
            if animation_type != AnimationType.NONE:
                css += self.get_animation_css(animation_type, category)
        
        return css
    
    def apply_animation(self, element_id: str, animation_type: AnimationType, category: str) -> str:
        """Generate JavaScript to apply animation to an element."""
        if animation_type == AnimationType.NONE:
            return ""
        
        if category not in self.animations or animation_type not in self.animations[category]:
            return ""
        
        class_name = f"{category}-{animation_type.value}"
        
        js = f"""
        (function() {{
            const element = document.getElementById('{element_id}');
            if (element) {{
                element.classList.add('{class_name}');
            }}
        }})();
        """
        
        return js
    
    def create_loading_spinner(self, spinner_type: str = "circle") -> str:
        """Create HTML for a loading spinner with the theme's loading animation."""
        animation_class = f"loading-{self.theme.animations.loading.value}"
        
        if spinner_type == "circle":
            spinner_html = f"""
            <div class="loading-spinner {animation_class}">
                <svg width="50" height="50" viewBox="0 0 50 50">
                    <circle cx="25" cy="25" r="20" fill="none" stroke="var(--primary-color)" stroke-width="4" stroke-dasharray="60 20" />
                </svg>
            </div>
            """
        elif spinner_type == "dots":
            spinner_html = f"""
            <div class="loading-dots {animation_class}">
                <div class="dot"></div>
                <div class="dot"></div>
                <div class="dot"></div>
            </div>
            """
        elif spinner_type == "bar":
            spinner_html = f"""
            <div class="loading-bar-container">
                <div class="loading-bar {animation_class}"></div>
            </div>
            """
        else:
            spinner_html = f"""
            <div class="loading-default {animation_class}">Loading...</div>
            """
        
        return spinner_html

class CSSGenerator:
    """Generator for CSS based on theme."""
    
    def __init__(self, theme: UITheme):
        self.theme = theme
        self.animation_controller = AnimationController(theme)
    
    def generate_root_variables(self) -> str:
        """Generate CSS root variables from theme."""
        colors = self.theme.colors
        color_variants = colors.generate_variants()
        
        # Convert hex colors to RGB values for use in rgba()
        rgb_values = {}
        for name, color in colors.__dict__.items():
            if name.startswith("_") or not isinstance(color, str) or not color.startswith("#"):
                continue
            
            # Skip rgba colors
            if "rgba" in color:
                continue
            
            # Convert hex to RGB
            r = int(color[1:3], 16)
            g = int(color[3:5], 16)
            b = int(color[5:7], 16)
            rgb_values[f"{name}-rgb"] = f"{r}, {g}, {b}"
        
        # Build CSS variables
        css = ":root {\n"
        
        # Add color variables
        for name, color in colors.__dict__.items():
            if name.startswith("_") or not isinstance(color, str):
                continue
            css += f"    --{name.replace('_', '-')}: {color};\n"
        
        # Add RGB values
        for name, value in rgb_values.items():
            css += f"    --{name.replace('_', '-')}: {value};\n"
        
        # Add color variants
        for name, color in color_variants.items():
            css += f"    --{name.replace('_', '-')}: {color};\n"
        
        # Add theme properties
        css += f"    --border-radius: {self.theme.border_radius};\n"
        css += f"    --shadow-intensity: {self.theme.shadow_intensity};\n"
        css += f"    --glass-opacity: {self.theme.glass_opacity};\n"
        css += f"    --glass-blur: {self.theme.glass_blur};\n"
        css += f"    --border-width: {self.theme.border_width};\n"
        
        css += "}\n"
        return css
    
    def generate_font_imports(self) -> str:
        """Generate CSS for font imports."""
        return """
        @import url('https://fonts.googleapis.com/css2?family=Orbitron:wght@400;500;600;700&family=Rajdhani:wght@300;400;500;600;700&family=Roboto:wght@300;400;500;700&family=Roboto+Condensed:wght@300;400;700&family=Roboto+Mono:wght@300;400;500;700&family=Share+Tech+Mono&family=VT323&display=swap');
        """
    
    def generate_font_styles(self) -> str:
        """Generate CSS for font styles."""
        fonts = self.theme.fonts
        
        css = """
        /* Typography */
        body, .streamlit-container {
            font-family: """ + fonts.primary + """;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: """ + fonts.heading + """;
            font-weight: 700;
            letter-spacing: 1px;
        }
        
        .futuristic-header {
            font-family: """ + fonts.display + """;
            font-weight: 700;
            letter-spacing: 2px;
            background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-transform: uppercase;
            margin-bottom: 30px;
        }
        
        code, pre, .terminal {
            font-family: """ + fonts.monospace + """;
        }
        
        .secondary-text {
            font-family: """ + fonts.secondary + """;
        }
        """
        
        return css
    
    def generate_base_styles(self) -> str:
        """Generate base CSS styles."""
        return """
        /* Global styles */
        body {
            background-color: var(--background);
            color: var(--text);
            background-image: 
                radial-gradient(circle at 10% 20%, rgba(var(--primary-rgb), 0.1) 0%, transparent 20%),
                radial-gradient(circle at 90% 80%, rgba(var(--secondary-rgb), 0.1) 0%, transparent 20%);
            background-attachment: fixed;
        }
        
        /* Glass morphism effect */
        .glass-card {
            background: var(--card-bg);
            backdrop-filter: blur(var(--glass-blur));
            -webkit-backdrop-filter: blur(var(--glass-blur));
            border-radius: var(--border-radius);
            border: var(--border-width) solid rgba(255, 255, 255, 0.18);
            box-shadow: 0 8px 32px 0 rgba(0, 0, 0, calc(0.37 * var(--shadow-intensity)));
            padding: 20px;
            margin-bottom: 20px;
        }
        
        /* Gradient borders */
        .gradient-border {
            position: relative;
            border-radius: var(--border-radius);
            padding: 20px;
        }
        
        .gradient-border::before {
            content: "";
            position: absolute;
            top: 0;
            left: 0;
            right: 0;
            bottom: 0;
            border-radius: var(--border-radius);
            padding: 2px;
            background: linear-gradient(45deg, var(--gradient-start), var(--gradient-end));
            -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
            -webkit-mask-composite: xor;
            mask-composite: exclude;
            pointer-events: none;
        }
        
        /* Terminal styles */
        .terminal {
            background-color: rgba(10, 10, 31, 0.9);
            color: var(--success);
            font-family: """ + self.theme.fonts.monospace + """;
            padding: 15px;
            border-radius: var(--border-radius);
            border: 1px solid var(--success);
            height: 300px;
            overflow-y: auto;
        }
        
        .terminal-prompt {
            color: var(--success);
        }
        
        .terminal-output {
            color: var(--text);
        }
        
        /* Status indicators */
        .status-indicator {
            display: inline-block;
            width: 10px;
            height: 10px;
            border-radius: 50%;
            margin-right: 5px;
        }
        
        .status-active {
            background-color: var(--success);
            box-shadow: 0 0 10px var(--success);
        }
        
        .status-idle {
            background-color: var(--warning);
            box-shadow: 0 0 10px var(--warning);
        }
        
        .status-error {
            background-color: var(--error);
            box-shadow: 0 0 10px var(--error);
        }
        
        /* Streamlit elements customization */
        .stTextInput > div > div > input {
            background-color: rgba(var(--background-rgb), 0.5);
            color: var(--text);
            border: 1px solid rgba(255, 255, 255, 0.18);
            border-radius: var(--border-radius);
        }
        
        .stButton > button {
            background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
            color: white;
            border: none;
            border-radius: var(--border-radius);
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px);
            box-shadow: 0 5px 15px rgba(var(--primary-rgb), 0.4);
        }
        
        /* Chat message styling */
        .user-message {
            background-color: rgba(var(--accent-rgb), 0.1);
            border-left: 3px solid var(--accent);
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 0 5px 5px 0;
        }
        
        .ai-message {
            background-color: rgba(var(--success-rgb), 0.1);
            border-left: 3px solid var(--success);
            padding: 10px;
            margin-bottom: 10px;
            border-radius: 0 5px 5px 0;
        }
        
        /* Custom scrollbar */
        ::-webkit-scrollbar {
            width: 8px;
            height: 8px;
        }
        
        ::-webkit-scrollbar-track {
            background: rgba(20, 20, 40, 0.5);
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb {
            background: linear-gradient(var(--gradient-start), var(--gradient-end));
            border-radius: 4px;
        }
        
        ::-webkit-scrollbar-thumb:hover {
            background: linear-gradient(var(--gradient-end), var(--gradient-start));
        }
        
        /* Metric cards */
        .metric-card {
            background: var(--card-bg);
            backdrop-filter: blur(var(--glass-blur));
            border-radius: var(--border-radius);
            border: var(--border-width) solid rgba(255, 255, 255, 0.18);
            padding: 15px;
            text-align: center;
            transition: all 0.3s ease;
        }
        
        .metric-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 10px 20px rgba(0, 0, 0, 0.2);
        }
        
        .metric-value {
            font-size: 24px;
            font-weight: bold;
            color: var(--accent);
        }
        
        .metric-label {
            font-size: 14px;
            color: var(--text);
            opacity: 0.8;
        }
        
        /* Loading spinners */
        .loading-spinner svg {
            transform-origin: center;
        }
        
        .loading-dots {
            display: flex;
            justify-content: center;
            align-items: center;
            gap: 8px;
        }
        
        .loading-dots .dot {
            width: 10px;
            height: 10px;
            background-color: var(--primary);
            border-radius: 50%;
        }
        
        .loading-bar-container {
            width: 100%;
            height: 4px;
            background: rgba(var(--primary-rgb), 0.2);
            border-radius: 2px;
            overflow: hidden;
        }
        
        .loading-bar {
            height: 100%;
            width: 30%;
            background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
            border-radius: 2px;
            animation: loading-bar 2s infinite ease-in-out;
        }
        
        @keyframes loading-bar {
            0% { transform: translateX(-100%); }
            100% { transform: translateX(400%); }
        }
        """
    
    def generate_theme_specific_styles(self) -> str:
        """Generate theme-specific CSS styles."""
        theme_type = self.theme.type
        css = ""
        
        if theme_type == ThemeType.DARK_GLASS:
            css += """
            /* Dark Glass theme specific styles */
            .streamlit-container {
                background: linear-gradient(135deg, rgba(10, 10, 31, 0.7), rgba(20, 20, 40, 0.7));
            }
            
            .sidebar .sidebar-content {
                background-color: rgba(10, 10, 31, 0.8);
                backdrop-filter: blur(10px);
            }
            """
        elif theme_type == ThemeType.CYBERPUNK:
            css += """
            /* Cyberpunk theme specific styles */
            body::before {
                content: "";
                position: fixed;
                top: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end));
                z-index: 1000;
            }
            
            body::after {
                content: "";
                position: fixed;
                bottom: 0;
                left: 0;
                right: 0;
                height: 2px;
                background: linear-gradient(90deg, var(--gradient-end), var(--gradient-start));
                z-index: 1000;
            }
            
            .futuristic-header {
                position: relative;
                text-shadow: 0 0 5px var(--primary);
            }
            
            .futuristic-header::before {
                content: attr(data-text);
                position: absolute;
                left: 2px;
                text-shadow: -1px 0 var(--secondary);
                top: 0;
                color: var(--primary);
                overflow: hidden;
                clip: rect(0, 900px, 0, 0);
                animation: cyberpunk-glitch 3s infinite linear alternate-reverse;
            }
            
            @keyframes cyberpunk-glitch {
                0% { clip: rect(0, 900px, 0, 0); }
                5% { clip: rect(0, 900px, 0, 0); }
                10% { clip: rect(0, 900px, 30px, 0); }
                15% { clip: rect(0, 900px, 0, 0); }
                20% { clip: rect(0, 900px, 10px, 0); }
                25% { clip: rect(0, 900px, 0, 0); }
                30% { clip: rect(0, 900px, 0, 0); }
                35% { clip: rect(0, 900px, 20px, 0); }
                40% { clip: rect(0, 900px, 0, 0); }
                45% { clip: rect(0, 900px, 0, 0); }
                50% { clip: rect(0, 900px, 5px, 0); }
                55% { clip: rect(0, 900px, 0, 0); }
                60% { clip: rect(0, 900px, 0, 0); }
                65% { clip: rect(0, 900px, 0, 0); }
                70% { clip: rect(0, 900px, 15px, 0); }
                75% { clip: rect(0, 900px, 0, 0); }
                80% { clip: rect(0, 900px, 0, 0); }
                85% { clip: rect(0, 900px, 25px, 0); }
                90% { clip: rect(0, 900px, 0, 0); }
                95% { clip: rect(0, 900px, 0, 0); }
                100% { clip: rect(0, 900px, 0, 0); }
            }
            """
        elif theme_type == ThemeType.HOLOGRAPHIC:
            css += """
            /* Holographic theme specific styles */
            .glass-card {
                background: linear-gradient(135deg, 
                    rgba(var(--primary-rgb), 0.2), 
                    rgba(var(--secondary-rgb), 0.2));
                border: 1px solid rgba(255, 255, 255, 0.2);
                box-shadow: 
                    0 8px 32px 0 rgba(0, 0, 0, 0.2),
                    0 0 10px rgba(var(--primary-rgb), 0.3),
                    0 0 20px rgba(var(--secondary-rgb), 0.2);
            }
            
            .glass-card::before {
                content: "";
                position: absolute;
                top: 0;
                left: 0;
                right: 0;
                bottom: 0;
                background: linear-gradient(135deg, 
                    rgba(var(--primary-rgb), 0.1) 0%, 
                    rgba(255, 255, 255, 0.05) 50%, 
                    rgba(var(--secondary-rgb), 0.1) 100%);
                opacity: 0.5;
                z-index: -1;
                border-radius: var(--border-radius);
                animation: holographic-shift 10s infinite linear;
            }
            
            @keyframes holographic-shift {
                0% { background-position: 0% 0%; }
                50% { background-position: 100% 100%; }
                100% { background-position: 0% 0%; }
            }
            """
        
        return css
    
    def generate_animation_styles(self) -> str:
        """Generate CSS for animations."""
        return self.animation_controller.get_all_animations_css()
    
    def generate_custom_styles(self) -> str:
        """Generate custom CSS from theme."""
        return self.theme.custom_css
    
    def generate_complete_css(self) -> str:
        """Generate complete CSS for the theme."""
        css_parts = [
            self.generate_font_imports(),
            self.generate_root_variables(),
            self.generate_font_styles(),
            self.generate_base_styles(),
            self.generate_theme_specific_styles(),
            self.generate_animation_styles(),
            self.generate_custom_styles()
        ]
        
        return "\n\n".join(css_parts)
    
    def save_css_to_file(self, filepath: Path) -> None:
        """Save generated CSS to file."""
        css = self.generate_complete_css()
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(css)

class ThemeManager:
    """Manager for UI themes."""
    
    def __init__(self, themes_dir: Path = THEMES_DIR):
        self.themes_dir = themes_dir
        self.themes: Dict[str, UITheme] = {}
        self.current_theme: Optional[UITheme] = None
        self.css_generator: Optional[CSSGenerator] = None
        
        # Create themes directory if it doesn't exist
        self.themes_dir.mkdir(parents=True, exist_ok=True)
        
        # Load built-in themes
        self.initialize_built_in_themes()
        
        # Load custom themes from directory
        self.load_themes()
    
    def initialize_built_in_themes(self) -> None:
        """Initialize built-in themes."""
        # Dark Glass theme (default)
        dark_glass = UITheme(
            name="Dark Glass",
            type=ThemeType.DARK_GLASS,
            colors=ColorScheme.dark_glass(),
            fonts=FontSet.ocr(),
            animations=AnimationSet.default(),
            glass_opacity=0.7,
            glass_blur="10px"
        )
        self.themes[dark_glass.name.lower()] = dark_glass
        
        # Light Glass theme
        light_glass = UITheme(
            name="Light Glass",
            type=ThemeType.LIGHT_GLASS,
            colors=ColorScheme.light_glass(),
            fonts=FontSet.clean(),
            animations=AnimationSet.default(),
            glass_opacity=0.7,
            glass_blur="10px"
        )
        self.themes[light_glass.name.lower()] = light_glass
        
        # Cyberpunk theme
        cyberpunk = UITheme(
            name="Cyberpunk",
            type=ThemeType.CYBERPUNK,
            colors=ColorScheme.cyberpunk(),
            fonts=FontSet.futuristic(),
            animations=AnimationSet.cyberpunk(),
            glass_opacity=0.8,
            glass_blur="5px",
            border_width="2px"
        )
        self.themes[cyberpunk.name.lower()] = cyberpunk
        
        # Terminal theme
        terminal = UITheme(
            name="Terminal",
            type=ThemeType.TERMINAL,
            colors=ColorScheme.terminal(),
            fonts=FontSet.terminal(),
            animations=AnimationSet.minimal(),
            glass_opacity=0.9,
            glass_blur="0px",
            border_width="1px"
        )
        self.themes[terminal.name.lower()] = terminal
        
        # Holographic theme
        holographic = UITheme(
            name="Holographic",
            type=ThemeType.HOLOGRAPHIC,
            colors=ColorScheme.holographic(),
            fonts=FontSet.futuristic(),
            animations=AnimationSet.holographic(),
            glass_opacity=0.5,
            glass_blur="15px",
            border_width="1px",
            shadow_intensity=1.5
        )
        self.themes[holographic.name.lower()] = holographic
        
        # Quantum theme
        quantum = UITheme(
            name="Quantum",
            type=ThemeType.QUANTUM,
            colors=ColorScheme.quantum(),
            fonts=FontSet.futuristic(),
            animations=AnimationSet.default(),
            glass_opacity=0.6,
            glass_blur="12px",
            border_width="1px",
            shadow_intensity=1.2
        )
        self.themes[quantum.name.lower()] = quantum
        
        # Save built-in themes
        for theme in self.themes.values():
            theme.save(self.themes_dir)
    
    def load_themes(self) -> None:
        """Load themes from the themes directory."""
        if not self.themes_dir.exists():
            return
        
        for filepath in self.themes_dir.glob("*.json"):
            try:
                theme = UITheme.load(filepath)
                self.themes[theme.name.lower()] = theme
            except Exception as e:
                print(f"Error loading theme from {filepath}: {e}")
    
    def get_theme(self, name: str) -> Optional[UITheme]:
        """Get a theme by name."""
        return self.themes.get(name.lower())
    
    def set_theme(self, name: str) -> bool:
        """Set the current theme by name."""
        theme = self.get_theme(name.lower())
        if not theme:
            return False
        
        self.current_theme = theme
        self.css_generator = CSSGenerator(theme)
        return True
    
    def get_current_theme(self) -> Optional[UITheme]:
        """Get the current theme."""
        return self.current_theme
    
    def add_theme(self, theme: UITheme) -> None:
        """Add a new theme."""
        self.themes[theme.name.lower()] = theme
        theme.save(self.themes_dir)
    
    def delete_theme(self, name: str) -> bool:
        """Delete a theme."""
        theme = self.get_theme(name.lower())
        if not theme:
            return False
        
        # Don't delete built-in themes
        if theme.type in [ThemeType.DARK_GLASS, ThemeType.LIGHT_GLASS, ThemeType.CYBERPUNK, 
                         ThemeType.TERMINAL, ThemeType.HOLOGRAPHIC, ThemeType.QUANTUM]:
            return False
        
        # Remove from memory
        del self.themes[name.lower()]
        
        # Remove file
        filepath = self.themes_dir / f"{name.lower().replace(' ', '_')}.json"
        if filepath.exists():
            filepath.unlink()
        
        return True
    
    def get_theme_names(self) -> List[str]:
        """Get a list of all theme names."""
        return list(self.themes.keys())
    
    def apply_theme_to_streamlit(self) -> None:
        """Apply the current theme to Streamlit."""
        if not self.current_theme or not self.css_generator:
            return
        
        # Generate CSS
        css = self.css_generator.generate_complete_css()
        
        # Apply CSS using st.markdown
        st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)
    
    def create_theme_preview(self, theme: UITheme) -> str:
        """Create HTML preview of a theme."""
        css_generator = CSSGenerator(theme)
        css = css_generator.generate_complete_css()
        
        html = f"""
        <html>
        <head>
            <title>{theme.name} Preview</title>
            <style>{css}</style>
        </head>
        <body>
            <div class="glass-card">
                <h1 class="futuristic-header">{theme.name}</h1>
                <p>This is a preview of the {theme.name} theme.</p>
                <div class="gradient-border">
                    <p>This is a gradient border element.</p>
                </div>
                <div class="status-indicator status-active"></div> Active
                <div class="status-indicator status-idle"></div> Idle
                <div class="status-indicator status-error"></div> Error
            </div>
            
            <div class="terminal">
                <div class="terminal-prompt">$ python app.py</div>
                <div class="terminal-output">Starting Skyscope Sentinel Intelligence...</div>
                <div class="terminal-output">Initializing agent swarm...</div>
                <div class="terminal-prompt">$</div>
            </div>
            
            <div class="glass-card">
                <h3>Loading Indicators</h3>
                <div class="loading-spinner loading-pulse">
                    <svg width="50" height="50" viewBox="0 0 50 50">
                        <circle cx="25" cy="25" r="20" fill="none" stroke="var(--primary-color)" stroke-width="4" stroke-dasharray="60 20" />
                    </svg>
                </div>
                
                <div class="loading-bar-container">
                    <div class="loading-bar loading-pulse"></div>
                </div>
            </div>
        </body>
        </html>
        """
        
        return html
    
    def save_theme_preview(self, theme: UITheme, filepath: Path) -> None:
        """Save theme preview to file."""
        html = self.create_theme_preview(theme)
        
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            f.write(html)

# Initialize theme manager as a singleton
_theme_manager = None

def get_theme_manager() -> ThemeManager:
    """Get the theme manager singleton."""
    global _theme_manager
    if _theme_manager is None:
        _theme_manager = ThemeManager()
    return _theme_manager

def apply_theme(theme_name: str = "Dark Glass") -> None:
    """Apply a theme to the Streamlit app."""
    theme_manager = get_theme_manager()
    if theme_manager.set_theme(theme_name):
        theme_manager.apply_theme_to_streamlit()

def get_animation_controller() -> AnimationController:
    """Get the animation controller for the current theme."""
    theme_manager = get_theme_manager()
    current_theme = theme_manager.get_current_theme()
    if not current_theme:
        # Use default theme if none is set
        theme_manager.set_theme("Dark Glass")
        current_theme = theme_manager.get_current_theme()
    
    return AnimationController(current_theme)

def create_loading_spinner(spinner_type: str = "circle") -> str:
    """Create HTML for a loading spinner."""
    animation_controller = get_animation_controller()
    return animation_controller.create_loading_spinner(spinner_type)

def create_progress_bar(progress: float, label: str = "") -> str:
    """Create HTML for a progress bar."""
    theme_manager = get_theme_manager()
    current_theme = theme_manager.get_current_theme()
    if not current_theme:
        theme_manager.set_theme("Dark Glass")
        current_theme = theme_manager.get_current_theme()
    
    # Ensure progress is between 0 and 1
    progress = max(0, min(1, progress))
    
    # Create progress bar HTML
    html = f"""
    <div class="progress-bar-container" style="width: 100%; height: 8px; background: rgba(var(--primary-rgb), 0.2); border-radius: 4px; overflow: hidden; margin: 10px 0;">
        <div class="progress-bar background-gradient-shift" style="height: 100%; width: {progress * 100}%; background: linear-gradient(90deg, var(--gradient-start), var(--gradient-end)); border-radius: 4px;">
        </div>
    </div>
    """
    
    if label:
        html += f"""
        <div style="font-size: 12px; color: var(--text); opacity: 0.8; text-align: center; margin-top: 5px;">
            {label}: {progress * 100:.1f}%
        </div>
        """
    
    return html

def create_thinking_animation(text: str = "Thinking") -> str:
    """Create HTML for a thinking animation."""
    theme_manager = get_theme_manager()
    current_theme = theme_manager.get_current_theme()
    if not current_theme:
        theme_manager.set_theme("Dark Glass")
        current_theme = theme_manager.get_current_theme()
    
    # Create thinking animation HTML
    html = f"""
    <div class="thinking-container" style="display: flex; align-items: center; margin: 10px 0;">
        <div style="font-family: {current_theme.fonts.monospace}; color: var(--accent); margin-right: 10px;">
            {text}
        </div>
        <div class="thinking-dots loading-pulse" style="display: flex;">
            <div style="width: 6px; height: 6px; border-radius: 50%; background-color: var(--accent); margin-right: 4px;"></div>
            <div style="width: 6px; height: 6px; border-radius: 50%; background-color: var(--accent); margin-right: 4px;"></div>
            <div style="width: 6px; height: 6px; border-radius: 50%; background-color: var(--accent);"></div>
        </div>
    </div>
    """
    
    return html

def create_ocr_text(text: str, animate: bool = True) -> str:
    """Create HTML for OCR-style text."""
    animation_class = "text-typing" if animate else ""
    
    html = f"""
    <div class="ocr-text {animation_class}" style="font-family: 'OCR A Extended', 'Share Tech Mono', monospace; letter-spacing: 1px;">
        {text}
    </div>
    """
    
    return html

def create_glitch_text(text: str) -> str:
    """Create HTML for glitch text effect."""
    html = f"""
    <div class="glitch-text" data-text="{text}" style="position: relative; font-family: 'Orbitron', sans-serif; font-weight: bold; color: var(--text);">
        {text}
    </div>
    <style>
        .glitch-text {{
            position: relative;
        }}
        
        .glitch-text::before,
        .glitch-text::after {{
            content: attr(data-text);
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
        }}
        
        .glitch-text::before {{
            left: 2px;
            text-shadow: -1px 0 var(--secondary);
            background: var(--background);
            overflow: hidden;
            clip: rect(0, 900px, 0, 0);
            animation: glitch-effect-1 3s infinite linear alternate-reverse;
        }}
        
        .glitch-text::after {{
            left: -2px;
            text-shadow: -1px 0 var(--primary);
            background: var(--background);
            overflow: hidden;
            clip: rect(0, 900px, 0, 0);
            animation: glitch-effect-2 2s infinite linear alternate-reverse;
        }}
        
        @keyframes glitch-effect-1 {{
            0% {{ clip: rect(0, 900px, 0, 0); }}
            5% {{ clip: rect(0, 900px, 0, 0); }}
            10% {{ clip: rect(0, 900px, 30px, 0); }}
            15% {{ clip: rect(0, 900px, 0, 0); }}
            20% {{ clip: rect(0, 900px, 10px, 0); }}
            25% {{ clip: rect(0, 900px, 0, 0); }}
            30% {{ clip: rect(0, 900px, 0, 0); }}
            35% {{ clip: rect(0, 900px, 20px, 0); }}
            40% {{ clip: rect(0, 900px, 0, 0); }}
            45% {{ clip: rect(0, 900px, 0, 0); }}
            50% {{ clip: rect(0, 900px, 5px, 0); }}
            55% {{ clip: rect(0, 900px, 0, 0); }}
            60% {{ clip: rect(0, 900px, 0, 0); }}
            65% {{ clip: rect(0, 900px, 0, 0); }}
            70% {{ clip: rect(0, 900px, 15px, 0); }}
            75% {{ clip: rect(0, 900px, 0, 0); }}
            80% {{ clip: rect(0, 900px, 0, 0); }}
            85% {{ clip: rect(0, 900px, 25px, 0); }}
            90% {{ clip: rect(0, 900px, 0, 0); }}
            95% {{ clip: rect(0, 900px, 0, 0); }}
            100% {{ clip: rect(0, 900px, 0, 0); }}
        }}
        
        @keyframes glitch-effect-2 {{
            0% {{ clip: rect(0, 900px, 0, 0); }}
            15% {{ clip: rect(0, 900px, 0, 0); }}
            20% {{ clip: rect(0, 900px, 30px, 0); }}
            25% {{ clip: rect(0, 900px, 0, 0); }}
            30% {{ clip: rect(0, 900px, 10px, 0); }}
            35% {{ clip: rect(0, 900px, 0, 0); }}
            40% {{ clip: rect(0, 900px, 0, 0); }}
            45% {{ clip: rect(0, 900px, 20px, 0); }}
            50% {{ clip: rect(0, 900px, 0, 0); }}
            55% {{ clip: rect(0, 900px, 0, 0); }}
            60% {{ clip: rect(0, 900px, 5px, 0); }}
            65% {{ clip: rect(0, 900px, 0, 0); }}
            70% {{ clip: rect(0, 900px, 0, 0); }}
            75% {{ clip: rect(0, 900px, 0, 0); }}
            80% {{ clip: rect(0, 900px, 15px, 0); }}
            85% {{ clip: rect(0, 900px, 0, 0); }}
            90% {{ clip: rect(0, 900px, 0, 0); }}
            95% {{ clip: rect(0, 900px, 25px, 0); }}
            100% {{ clip: rect(0, 900px, 0, 0); }}
        }}
    </style>
    """
    
    return html

def create_hologram_text(text: str) -> str:
    """Create HTML for hologram text effect."""
    html = f"""
    <div class="hologram-text" style="font-family: 'Orbitron', sans-serif; font-weight: bold; color: var(--primary); text-shadow: 0 0 5px var(--primary), 0 0 10px var(--primary); position: relative;">
        {text}
        <div class="hologram-scanline"></div>
    </div>
    <style>
        .hologram-text {{
            position: relative;
            display: inline-block;
            animation: hologram-flicker 4s infinite;
        }}
        
        .hologram-scanline {{
            position: absolute;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(to bottom, transparent, rgba(var(--primary-rgb), 0.2) 50%, transparent);
            background-size: 100% 4px;
            z-index: 1;
            animation: hologram-scanline 6s linear infinite;
            pointer-events: none;
        }}
        
        @keyframes hologram-flicker {{
            0% {{ opacity: 1; }}
            3% {{ opacity: 0.8; }}
            6% {{ opacity: 1; }}
            7% {{ opacity: 0.4; }}
            8% {{ opacity: 1; }}
            9% {{ opacity: 0.8; }}
            10% {{ opacity: 1; }}
            99% {{ opacity: 1; }}
            100% {{ opacity: 0.8; }}
        }}
        
        @keyframes hologram-scanline {{
            0% {{ background-position: 0 -100%; }}
            100% {{ background-position: 0 200%; }}
        }}
    </style>
    """
    
    return html
