const { sone, Text, Column, Span, Font } = require('sone');
const express = require('express');
const path = require('path');

const FONTS = {
  kantumruy: 'KantumruyPro-Regular.ttf',
  moul: 'Moul-Regular.ttf',
  battambang: 'Battambang-Regular.ttf',
  bayon: 'Bayon-Regular.ttf',
  notosans: 'NotoSansKhmer-Regular.ttf',
  siemreap: 'Siemreap-Regular.ttf',
  arial: 'Arial.ttf',
};

const LATIN_FONT = 'NotoSans';
const FONT_NAMES = Object.keys(FONTS);
const fontsDir = path.join(__dirname, 'fonts');
let fontsLoaded = false;

async function loadFonts() {
  if (fontsLoaded) return;
  for (const [name, file] of Object.entries(FONTS)) {
    await Font.load(name, path.join(fontsDir, file));
  }
  await Font.load(LATIN_FONT, path.join(fontsDir, 'NotoSans-Regular.ttf'));
  await Font.load('timesnewroman', path.join(fontsDir, 'Times New Roman.ttf'));
  fontsLoaded = true;
}

function splitKhmer(text) {
  const parts = [];
  const khmer = /[\u1780-\u17FF\u19E0-\u19FF]+/g;
  let last = 0;
  for (const m of text.matchAll(khmer)) {
    if (m.index > last) parts.push({ text: text.slice(last, m.index), isKhmer: false });
    parts.push({ text: m[0], isKhmer: true });
    last = m.index + m[0].length;
  }
  if (last < text.length) parts.push({ text: text.slice(last), isKhmer: false });
  return parts;
}

async function renderWithFallback(text, fontFamily, fontSize, color, background, padding, paddingTop, paddingRight, paddingBottom, paddingLeft) {
  await loadFonts();
  const parts = splitKhmer(text);
  const hasKhmer = parts.some(p => p.isKhmer);

  // Determine the latin font to use for this request
  let latinFont;
  if (fontFamily === 'arial') {
    latinFont = 'arial';
  } else if (fontFamily === 'kantumruy' || fontFamily === 'notosans') {
    latinFont = fontFamily;
  } else {
    latinFont = 'arial';
  }

  let textNode;
  if (hasKhmer && parts.length > 1) {
    const spans = parts.map(p =>
      p.isKhmer
        ? Span(p.text).font(fontFamily)
        : Span(p.text).font(latinFont)
    );
    textNode = Text(...spans).size(fontSize).color(color);
  } else if (!hasKhmer) {
    textNode = Text(text).font(latinFont).size(fontSize).color(color);
  } else {
    textNode = Text(text).font(fontFamily).size(fontSize).color(color);
  }

  return Column(textNode)
    .paddingTop(paddingTop ?? padding)
    .paddingRight(paddingRight ?? padding)
    .paddingBottom(paddingBottom ?? padding)
    .paddingLeft(paddingLeft ?? padding)
    .bg(background);
}

const app = express();
app.use(express.json({ limit: '5mb' }));

app.get('/fonts', (req, res) => {
  res.json(FONT_NAMES);
});

app.post('/render', async (req, res) => {
  try {
    const {
      text,
      font = 'notosans',
      fontSize = 48,
      color = '#000',
      background = '#fff',
      padding = 32,
      paddingTop,
      paddingRight,
      paddingBottom,
      paddingLeft,
      format = 'png',
    } = req.body;

    if (!text) {
      return res.status(400).json({ error: 'text is required' });
    }

    await loadFonts();
    const fontFamily = FONT_NAMES.includes(font) ? font : 'notosans';
    const node = await renderWithFallback(text, fontFamily, fontSize, color, background, padding, paddingTop, paddingRight, paddingBottom, paddingLeft);

    const result = await sone(node);
    let buffer;
    let contentType;

    switch (format) {
      case 'jpg':
      case 'jpeg':
        buffer = await result.jpg();
        contentType = 'image/jpeg';
        break;
      case 'webp':
        buffer = await result.webp();
        contentType = 'image/webp';
        break;
      case 'pdf':
        buffer = await result.pdf();
        contentType = 'application/pdf';
        break;
      default:
        buffer = await result.png();
        contentType = 'image/png';
        break;
    }

    res.setHeader('Content-Type', contentType);
    res.setHeader('X-Font', fontFamily);
    res.setHeader('X-FontSize', fontSize.toString());
    res.send(buffer);
  } catch (err) {
    console.error(err);
    res.status(500).json({ error: err.message });
  }
});

app.get('/', (req, res) => {
  res.send(`<!DOCTYPE html>
<html lang="km">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Khmer Text Renderer</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <link href="https://fonts.googleapis.com/css2?family=Noto+Sans+Khmer:wght@100..900&display=swap" rel="stylesheet">
  <style> body { font-family: 'Noto Sans Khmer', sans-serif; } </style>
</head>
<body class="bg-gray-100 min-h-screen flex items-center justify-center">
  <div class="bg-white rounded-xl shadow-lg p-8 w-full max-w-xl">
    <h1 class="text-2xl font-bold mb-6 text-center">Khmer Text Renderer</h1>
    <div class="space-y-4">
      <div>
        <label class="block text-sm font-medium mb-1">Khmer Text</label>
        <textarea id="text" rows="3" class="w-full border rounded-lg p-3 text-lg"
          placeholder="វាយអត្ថបទខ្មែរនៅទីនេះ...">សួស្តី ពិភពលោក</textarea>
      </div>
      <div class="grid grid-cols-3 gap-4">
        <div>
          <label class="block text-sm font-medium mb-1">Font</label>
          <select id="font" class="w-full border rounded-lg p-2">
            ${FONT_NAMES.map(f => '<option value="' + f + '">' + f + '</option>').join('')}
          </select>
        </div>
        <div>
          <label class="block text-sm font-medium mb-1">Size</label>
          <input id="fontSize" type="number" value="48" min="12" max="200"
            class="w-full border rounded-lg p-2">
        </div>
        <div>
          <label class="block text-sm font-medium mb-1">Format</label>
          <select id="format" class="w-full border rounded-lg p-2">
            <option value="png">PNG</option>
            <option value="jpg">JPG</option>
            <option value="webp">WebP</option>
            <option value="pdf">PDF</option>
          </select>
        </div>
      </div>
      <button onclick="doRender()"
        class="w-full bg-blue-600 text-white py-2 rounded-lg hover:bg-blue-700 font-medium">
        Render
      </button>
    </div>
    <div id="result" class="mt-6 text-center hidden">
      <img id="preview" class="mx-auto rounded-lg shadow max-w-full" />
      <a id="download" class="mt-3 inline-block text-blue-600 underline text-sm" download>Download</a>
    </div>
  </div>
  <script>
    async function doRender() {
      const text = document.getElementById('text').value;
      const font = document.getElementById('font').value;
      const fontSize = parseInt(document.getElementById('fontSize').value);
      const format = document.getElementById('format').value;

      const res = await fetch('/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ text, font, fontSize, format }),
      });

      const blob = await res.blob();
      const url = URL.createObjectURL(blob);
      document.getElementById('preview').src = url;
      document.getElementById('download').href = url;
      document.getElementById('download').download = 'khmer-text.' + format;
      document.getElementById('result').classList.remove('hidden');
    }
  </script>
</body>
</html>`);
});

const PORT = process.env.PORT || 3456;
app.listen(PORT, () => {
  console.log('Khmer Text Renderer running at http://localhost:' + PORT);
});
