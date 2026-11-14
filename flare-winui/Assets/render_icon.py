# SPDX-License-Identifier: Apache-2.0
"""Render the polygon SVG sources into native Windows icon frames with Pillow."""
from io import BytesIO
from pathlib import Path
import struct
import xml.etree.ElementTree as ET

from PIL import Image, ImageColor, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SIZES = (16, 20, 24, 32, 40, 48, 64, 128, 256)


def render(source, size):
    scale = size * 8 / 64
    image = Image.new("RGBA", (size * 8, size * 8))
    draw = ImageDraw.Draw(image)
    for element in ET.parse(source).getroot():
        if element.tag.rsplit("}", 1)[-1] != "polygon":
            continue
        points = [tuple(float(v) * scale for v in p.split(","))
                  for p in element.attrib["points"].split()]
        draw.polygon(points, fill=ImageColor.getrgb(element.attrib["fill"]))
        if "stroke" in element.attrib:
            draw.line(points + points[:1], fill=ImageColor.getrgb(element.attrib["stroke"]),
                      width=max(1, round(float(element.attrib["stroke-width"]) * scale)),
                      joint="curve")
    image = image.resize((size, size), Image.Resampling.LANCZOS)
    image.putalpha(image.getchannel("A").point(lambda alpha: 0 if alpha <= 2 else alpha))
    return image


def write_ico(frames):
    blobs = []
    for image in frames:
        output = BytesIO()
        image.save(output, "PNG")
        blobs.append(output.getvalue())
    offset = 6 + 16 * len(frames)
    entries = []
    for image, blob in zip(frames, blobs):
        size = image.width if image.width < 256 else 0
        entries.append(struct.pack("<BBBBHHII", size, size, 0, 0, 1, 32, len(blob), offset))
        offset += len(blob)
    (HERE / "flare-app.ico").write_bytes(
        struct.pack("<HHH", 0, 1, len(frames)) + b"".join(entries) + b"".join(blobs))


frames = [render(HERE / ("flare-app-small.svg" if size <= 32 else "flare-app.svg"), size)
          for size in SIZES]
for image in frames:
    image.save(HERE / f"flare-app-{image.width}.png")
render(HERE / "flare-app.svg", 512).save(HERE / "flare-app.png")
write_ico(frames)

sheet = Image.new("RGB", (1120, 620), "#101010")
draw = ImageDraw.Draw(sheet)
font = ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf", 15)
heading = ImageFont.truetype("C:/Windows/Fonts/seguisb.ttf", 26)
draw.text((32, 24), "Hyperflare", font=heading, fill="#F5F0E3")
draw.text((32, 65), "Windows app icon · vector mark and optical small-size frames", font=font, fill="#B0B0B0")
sheet.paste(frames[-1], (820, 28), frames[-1])
for y, background, ink in ((140, "#191919", "#B0B0B0"), (318, "#EEEEEE", "#555555")):
    draw.rounded_rectangle((24, y - 18, 790, y + 140), radius=4, fill=background)
    x = 48
    for image in frames[:-1]:
        sheet.paste(image, (x, y + (96 - image.height) // 2), image)
        draw.text((x, y + 112), str(image.width), font=font, fill=ink)
        x += image.width + 38
draw.text((32, 496), "Native sizes above: 16 / 20 / 24 / 32 / 40 / 48 / 64 / 128 px. ICO also includes 256 px.", font=font, fill="#B0B0B0")
small = frames[0].resize((96, 96), Image.Resampling.NEAREST)
sheet.paste(small, (838, 374), small)
draw.text((820, 488), "16 px · enlarged 6×", font=font, fill="#B0B0B0")
draw.text((32, 538), "Block lettering + diagonal solar-plume silhouette, adapted from the upstream banner.", font=font, fill="#B0B0B0")
draw.text((32, 570), "Cream #F5F0E3   /   NVIDIA green #76B900   /   charcoal #101010   /   transparent outer edge", font=font, fill="#B0B0B0")
sheet.save(HERE / "flare-app-preview.png")

small_sheet = Image.new("RGB", (760, 370), "#101010")
draw = ImageDraw.Draw(small_sheet)
draw.text((24, 18), "Taskbar / titlebar optical check", font=heading, fill="#F5F0E3")
for index, size in enumerate((16, 24, 32)):
    frame = frames[SIZES.index(size)]
    x = 40 + index * 240
    draw.text((x, 65), f"{size} px · actual / 6× pixels", font=font, fill="#B0B0B0")
    small_sheet.paste(frame, (x, 105), frame)
    expanded = frame.resize((size * 6, size * 6), Image.Resampling.NEAREST)
    small_sheet.paste(expanded, (x, 165), expanded)
small_sheet.save(HERE / "flare-app-small-preview.png")

with Image.open(HERE / "flare-app.ico") as icon:
    assert icon.ico.sizes() == {(size, size) for size in SIZES}
    for size in SIZES:
        frame = icon.ico.getimage((size, size)).convert("RGBA")
        assert frame.getpixel((0, 0))[3] == 0
        assert frame.getextrema()[3] == (0, 255)
print("Verified nine 32-bit icon frames with alpha:", ", ".join(map(str, SIZES)))
