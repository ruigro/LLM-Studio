#!/usr/bin/env python3
"""Create a proper Windows-compatible ICO file for the launcher"""
import sys
import struct
from pathlib import Path

def create_simple_ico(output_path):
    """
    Create a minimal but valid Windows ICO file.
    This creates a simple 32x32 icon with a rocket-like pattern.
    """
    # ICO header
    ico_header = struct.pack('<HHH', 0, 1, 1)  # Reserved, Type (1=ICO), Count (1 image)
    
    # Image directory entry for 32x32 image
    width = 32
    height = 32
    colors = 0  # 0 means more than 256 colors
    reserved = 0
    planes = 1
    bpp = 32  # 32 bits per pixel (RGBA)
    
    # Create a simple RGBA bitmap data for a rocket shape
    pixels = []
    for y in range(height):
        for x in range(width):
            # Simple rocket shape
            center_x, center_y = 16, 16
            dx, dy = x - center_x, y - center_y
            dist = (dx**2 + dy**2) ** 0.5
            
            if dist < 12:  # Main body
                r, g, b, a = 102, 126, 234, 255  # Purple
            elif dist < 14:  # Border
                r, g, b, a = 118, 75, 162, 255  # Darker purple
            else:
                r, g, b, a = 0, 0, 0, 0  # Transparent
            
            # Add flame at bottom
            if y > 22 and abs(dx) < 4:
                r, g, b, a = 255, 152, 0, 255  # Orange flame
            
            # Add nose cone at top
            if y < 10 and abs(dx) < (10 - y):
                r, g, b, a = 255, 107, 107, 255  # Red nose
                
            pixels.append(struct.pack('BBBB', b, g, r, a))  # BGR+A format
    
    bitmap_data = b''.join(pixels)
    
    # BMP info header (40 bytes)
    bmp_info_header = struct.pack(
        '<IIIHHIIIIII',
        40,  # Header size
        width,
        height * 2,  # Height * 2 for ICO format
        planes,
        bpp,
        0,  # Compression (0 = none)
        len(bitmap_data),
        0, 0,  # X/Y pixels per meter
        0, 0   # Colors used/important
    )
    
    image_data = bmp_info_header + bitmap_data
    
    # Image directory entry
    image_dir_entry = struct.pack(
        '<BBBBHHII',
        width,
        height,
        colors,
        reserved,
        planes,
        bpp,
        len(image_data),
        22  # Offset to image data (6 bytes header + 16 bytes dir entry)
    )
    
    # Write ICO file
    with open(output_path, 'wb') as f:
        f.write(ico_header)
        f.write(image_dir_entry)
        f.write(image_data)
    
    print(f"Created valid Windows ICO file: {output_path}")
    print(f"File size: {Path(output_path).stat().st_size} bytes")

if __name__ == "__main__":
    output = Path(__file__).parent / "rocket.ico"
    create_simple_ico(output)
    print("\nICO file created successfully!")
    print("This uses only Python standard library - no PIL/Pillow needed.")

