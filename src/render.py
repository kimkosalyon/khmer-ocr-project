import io
import freetype
import uharfbuzz as hb
from PIL import Image, ImageDraw


FONT_DIR = "fonts"
FONT_MAP = {
    "kantumruy": f"{FONT_DIR}/KantumruyPro-Regular.ttf",
    "moul": f"{FONT_DIR}/Moul-Regular.ttf",
    "battambang": f"{FONT_DIR}/Battambang-Regular.ttf",
    "bayon": f"{FONT_DIR}/Bayon-Regular.ttf",
    "notosans": f"{FONT_DIR}/NotoSansKhmer-Regular.ttf",
    "siemreap": f"{FONT_DIR}/Siemreap-Regular.ttf",
}


class KhmerTextRenderer:
    def __init__(self, font_path: str, font_size: int = 32):
        self.face = freetype.Face(font_path)
        self.face.set_char_size(font_size * 64)
        self.font_size = font_size

        blob = hb.Blob.from_file_path(font_path)
        face = hb.Face(blob)
        self.hb_font = hb.Font(face)
        self.hb_font.scale = (font_size * 64, font_size * 64)

        self.ascent = self.face.size.ascender >> 6
        self.descent = -(self.face.size.descender >> 6)

    def render(self, text: str, padding: int = 4) -> Image.Image:
        buf = hb.Buffer()
        buf.add_str(text)
        buf.guess_segment_properties()
        hb.shape(self.hb_font, buf, {})

        infos = buf.glyph_infos
        positions = buf.glyph_positions

        if not infos:
            return Image.new("L", (1, 1), 255)

        glyph_slots = []
        x_cursor = 0
        for info, pos in zip(infos, positions):
            self.face.load_glyph(info.codepoint, freetype.FT_LOAD_RENDER)
            bitmap = self.face.glyph.bitmap
            glyph_w = bitmap.width
            glyph_h = bitmap.rows if bitmap.rows > 0 else 1
            glyph_buf = bitmap.buffer

            img = Image.new("L", (glyph_w, glyph_h), 0)
            if glyph_buf:
                img = Image.open(io.BytesIO(glyph_buf)).convert("L") \
                    if False else Image.frombytes("L", (glyph_w, glyph_h), bytes(glyph_buf))

            x_offset = self.face.glyph.bitmap_left
            y_offset = self.ascent - self.face.glyph.bitmap_top

            x_advance = pos.x_advance >> 6
            y_advance = pos.y_advance >> 6

            x_pen = x_cursor + x_offset
            x_cursor += x_advance + (pos.x_offset >> 6)
            y_pen = y_offset + (pos.y_offset >> 6)

            glyph_slots.append((img, x_pen, y_pen))

        total_w = max((s[1] + s[0].width for s in glyph_slots), default=1) + padding * 2
        total_h = self.ascent + self.descent + padding * 2

        canvas = Image.new("L", (total_w, total_h), 255)
        for glyph_img, gx, gy in glyph_slots:
            canvas.paste(glyph_img, (gx + padding, gy + padding))

        return canvas


def list_fonts():
    return list(FONT_MAP.keys())


def get_font_path(name: str) -> str:
    return FONT_MAP[name]
