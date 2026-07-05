const { sone, Text, Column, Span, Font } = require('sone');
const path = require('path');
const fs = require('fs');

const FONTS = {
  kantumruy: 'KantumruyPro-Regular.ttf',
  moul: 'Moul-Regular.ttf',
  battambang: 'Battambang-Regular.ttf',
  bayon: 'Bayon-Regular.ttf',
  notosans: 'NotoSansKhmer-Regular.ttf',
  siemreap: 'Siemreap-Regular.ttf',
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
  await Font.load('arial', path.join(fontsDir, 'Arial.ttf'));
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

async function renderText(text, options = {}) {
  const {
    font = 'notosans',
    fontSize = 48,
    color = '#000',
    background = '#fff',
    padding = 32,
    paddingTop,
    paddingRight,
    paddingBottom,
    paddingLeft,
    width = undefined,
  } = options;

  await loadFonts();

  const fontFamily = FONT_NAMES.includes(font) ? font : 'notosans';

  // Determine the latin font to use for this request
  let latinFont;
  if (fontFamily === 'kantumruy' || fontFamily === 'notosans') {
    latinFont = fontFamily;
  } else {
    const englishFonts = ['arial', 'timesnewroman'];
    latinFont = englishFonts[Math.floor(Math.random() * englishFonts.length)];
  }

  const parts = splitKhmer(text);
  const hasKhmer = parts.some(p => p.isKhmer);

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

  const node = Column(textNode)
    .paddingTop(paddingTop ?? padding)
    .paddingRight(paddingRight ?? padding)
    .paddingBottom(paddingBottom ?? padding)
    .paddingLeft(paddingLeft ?? padding)
    .bg(background);
  if (width) node.width(width);

  return sone(node);
}

async function main() {
  const args = process.argv.slice(2);

  if (args.length === 0 || args.includes('--help') || args.includes('-h')) {
    console.log(`Khmer Text Renderer (Sone + HarfBuzz)

Usage: node render.js [options] <text>

Options:
  --font <name>          Font name (${FONT_NAMES.join(', ')}) [default: notosans]
  --size <px>            Font size in pixels [default: 48]
  --color <hex>          Text color [default: #000]
  --bg <hex>             Background color [default: #fff]
  --padding <px>         Padding all sides [default: 32]
  --padding-top <px>     Top padding (overrides --padding)
  --padding-right <px>   Right padding (overrides --padding)
  --padding-bottom <px>  Bottom padding (overrides --padding)
  --padding-left <px>    Left padding (overrides --padding)
  --output <path>        Output file [default: output.png]
  --format <type>        Output format (png, jpg, webp, pdf) [default: png]
  --list-fonts           List available fonts

Examples:
  node render.js "សួស្តី ពិភពលោក"
  node render.js --font battambang --size 64 "អត្ថបទខ្មែរ"
  node render.js --font moul --format jpg --output banner.jpg "បឋមកថា"
`);
    process.exit(0);
  }

  if (args.includes('--list-fonts')) {
    console.log('Available fonts:');
    for (const name of FONT_NAMES) {
      console.log('  ' + name + ' -> ' + FONTS[name]);
    }
    process.exit(0);
  }

  let font = 'notosans';
  let fontSize = 48;
  let color = '#000';
  let bg = '#fff';
  let padding = 32;
  let paddingTop, paddingRight, paddingBottom, paddingLeft;
  let output = 'output.png';
  let format = 'png';
  let text = '';

  for (let i = 0; i < args.length; i++) {
    switch (args[i]) {
      case '--font': font = args[++i]; break;
      case '--size': fontSize = parseInt(args[++i]); break;
      case '--color': color = args[++i]; break;
      case '--bg': bg = args[++i]; break;
      case '--padding': padding = parseInt(args[++i]); break;
      case '--padding-top': paddingTop = parseInt(args[++i]); break;
      case '--padding-right': paddingRight = parseInt(args[++i]); break;
      case '--padding-bottom': paddingBottom = parseInt(args[++i]); break;
      case '--padding-left': paddingLeft = parseInt(args[++i]); break;
      case '--output': output = args[++i]; break;
      case '--format': format = args[++i]; break;
      default: text = args[i]; break;
    }
  }

  if (!text) {
    console.error('Error: no text provided');
    process.exit(1);
  }

  const result = await renderText(text, { font, fontSize, color, background: bg, padding, paddingTop, paddingRight, paddingBottom, paddingLeft });
  const buffer = format === 'jpg' ? await result.jpg() : await result.png();
  fs.writeFileSync(output, buffer);
  console.log(`Saved ${output} (${buffer.length} bytes, font=${font}, size=${fontSize})`);
}

module.exports = { renderText, loadFonts, FONT_NAMES, FONTS };

if (require.main === module) {
  main().catch(console.error);
}
