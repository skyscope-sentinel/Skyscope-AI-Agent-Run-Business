#!/usr/bin/env python3
"""
Simple script to create a logo for the Skyscope application
"""

try:
    from PIL import Image, ImageDraw, ImageFont
    import math
    
    # Create a 512x512 logo
    size = 512
    img = Image.new('RGB', (size, size), (15, 17, 26))  # Dark background
    draw = ImageDraw.Draw(img)
    
    # Draw gradient circles
    center = size // 2
    colors = [(75, 94, 255), (138, 43, 226), (255, 20, 147)]
    
    for i, color in enumerate(colors):
        radius = 200 - i * 50
        draw.ellipse([center-radius, center-radius, center+radius, center+radius], 
                    outline=color, width=4)
    
    # Add connecting lines
    for angle in range(0, 360, 45):
        x1 = center + 80 * math.cos(math.radians(angle))
        y1 = center + 80 * math.sin(math.radians(angle))
        x2 = center + 120 * math.cos(math.radians(angle))
        y2 = center + 120 * math.sin(math.radians(angle))
        draw.line([(x1, y1), (x2, y2)], fill=(255, 255, 255), width=3)
    
    # Add center circle
    draw.ellipse([center-30, center-30, center+30, center+30], 
                fill=(75, 94, 255))
    
    # Save the logo
    img.save('logo.png', 'PNG')
    print('Logo created successfully')
    
except ImportError:
    print("PIL not available, creating a simple placeholder")
    # Create a simple text file as placeholder
    with open('logo.png', 'w') as f:
        f.write("# Placeholder logo file\n")
    print("Placeholder logo created")