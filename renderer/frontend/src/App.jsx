import { useState, useEffect } from 'react'

const FONTS = [
  'kantumruy', 'moul', 'battambang', 'bayon', 'notosans', 'siemreap',
]

// Helper to parse hex string to RGB
function hexToRgb(hex) {
  const c = hex.replace('#', '');
  if (c.length === 3) {
    const r = parseInt(c[0] + c[0], 16);
    const g = parseInt(c[1] + c[1], 16);
    const b = parseInt(c[2] + c[2], 16);
    return { r, g, b };
  }
  const r = parseInt(c.substring(0, 2), 16);
  const g = parseInt(c.substring(2, 4), 16);
  const b = parseInt(c.substring(4, 6), 16);
  return { r, g, b };
}

// Calculate simple luminance to check if color is light or dark
function isLight(hex) {
  try {
    const { r, g, b } = hexToRgb(hex);
    const y = 0.299 * r + 0.587 * g + 0.114 * b;
    return y >= 128;
  } catch (e) {
    return true; // Default fallback
  }
}

// Generate random color with controlled lightness
function getRandomColor(type) {
  const h = Math.floor(Math.random() * 360);
  const s = Math.floor(Math.random() * 90) + 10; // 10-100% saturation
  const l = type === 'light' ? Math.floor(Math.random() * 15) + 80 : Math.floor(Math.random() * 20); // 80-95% or 0-20%
  
  function hslToHex(h, s, l) {
    l /= 100;
    const a = s * Math.min(l, 1 - l) / 100;
    const f = n => {
      const k = (n + h / 30) % 12;
      const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
      return Math.round(255 * color).toString(16).padStart(2, '0');
    };
    return `#${f(0)}${f(8)}${f(4)}`;
  }
  return hslToHex(h, s, l);
}

// Generate random contrast-safe pair
function getRandomContrastPair(scheme = 'both') {
  const mode = scheme === 'both' ? (Math.random() > 0.5 ? 'light' : 'dark') : scheme;
  let bg_h, bg_s, bg_l;
  let tx_h, tx_s, tx_l;

  if (mode === 'light') {
    bg_h = Math.floor(Math.random() * 360);
    bg_s = Math.floor(Math.random() * 90) + 10; // 10-100% saturation
    bg_l = Math.floor(Math.random() * 18) + 80; // 80-98% lightness

    tx_h = Math.floor(Math.random() * 360);
    tx_s = Math.floor(Math.random() * 90) + 10; // 10-100% saturation
    tx_l = Math.floor(Math.random() * 25); // 0-25% lightness
  } else {
    bg_h = Math.floor(Math.random() * 360);
    bg_s = Math.floor(Math.random() * 90) + 10; // 10-100% saturation
    bg_l = Math.floor(Math.random() * 20); // 0-20% lightness

    tx_h = Math.floor(Math.random() * 360);
    tx_s = Math.floor(Math.random() * 90) + 10; // 10-100% saturation
    tx_l = Math.floor(Math.random() * 25) + 75; // 75-100% lightness
  }

  function hslToHex(h, s, l) {
    l /= 100;
    const a = s * Math.min(l, 1 - l) / 100;
    const f = n => {
      const k = (n + h / 30) % 12;
      const color = l - a * Math.max(Math.min(k - 3, 9 - k, 1), -1);
      return Math.round(255 * color).toString(16).padStart(2, '0');
    };
    return `#${f(0)}${f(8)}${f(4)}`;
  }

  return [hslToHex(tx_h, tx_s, tx_l), hslToHex(bg_h, bg_s, bg_l)];
}

function App() {
  const [text, setText] = useState('សួស្តី ពិភពលោក — ខ្មែរ OCR')
  const [font, setFont] = useState('notosans')
  const [fontSize, setFontSize] = useState(48)
  const [format, setFormat] = useState('png')
  const [color, setColor] = useState('#000000')
  const [background, setBackground] = useState('#ffffff')
  
  // Random color toggles
  const [randomText, setRandomText] = useState(false)
  const [randomBg, setRandomBg] = useState(false)

  const [paddingTop, setPaddingTop] = useState(32)
  const [paddingRight, setPaddingRight] = useState(32)
  const [paddingBottom, setPaddingBottom] = useState(32)
  const [paddingLeft, setPaddingLeft] = useState(32)
  const [imageUrl, setImageUrl] = useState(null)
  const [loading, setLoading] = useState(false)

  const shuffleColors = () => {
    if (randomText && randomBg) {
      const [newColor, newBg] = getRandomContrastPair('both');
      setColor(newColor);
      setBackground(newBg);
      return { newColor, newBg };
    } else if (randomText) {
      const bgLight = isLight(background);
      const newColor = getRandomColor(bgLight ? 'dark' : 'light');
      setColor(newColor);
      return { newColor, newBg: background };
    } else if (randomBg) {
      const textLight = isLight(color);
      const newBg = getRandomColor(textLight ? 'dark' : 'light');
      setBackground(newBg);
      return { newColor: color, newBg };
    }
    return { newColor: color, newBg: background };
  }

  const doRender = async () => {
    setLoading(true)
    try {
      // Dynamic color shuffle if toggles are enabled
      let reqColor = color;
      let reqBg = background;
      
      if (randomText || randomBg) {
        const shuffled = shuffleColors();
        reqColor = shuffled.newColor;
        reqBg = shuffled.newBg;
      }

      const res = await fetch('/render', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          text, font, fontSize, format,
          color: reqColor, background: reqBg,
          paddingTop, paddingRight, paddingBottom, paddingLeft,
        }),
      })
      const blob = await res.blob()
      const url = URL.createObjectURL(blob)
      setImageUrl(url)
    } catch (err) {
      console.error(err)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { doRender() }, [])

  return (
    <div className="min-h-screen bg-gray-100 flex items-center justify-center p-4">
      <div className="bg-white rounded-xl shadow-lg p-8 w-full max-w-xl">
        <h1 className="text-2xl font-bold mb-6 text-center text-gray-800">Khmer Text Renderer</h1>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium mb-1 text-gray-700">Khmer Text</label>
            <textarea
              value={text}
              onChange={e => setText(e.target.value)}
              rows={3}
              className="w-full border rounded-lg p-3 text-lg focus:ring-2 focus:ring-blue-500 focus:border-blue-500 outline-none"
              placeholder="វាយអត្ថបទខ្មែរនៅទីនេះ..."
            />
          </div>

          <div className="grid grid-cols-3 gap-4">
            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700">Font</label>
              <select value={font} onChange={e => setFont(e.target.value)}
                className="w-full border rounded-lg p-2 focus:ring-2 focus:ring-blue-500 outline-none">
                {FONTS.map(f => <option key={f} value={f}>{f}</option>)}
              </select>
            </div>
            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700">Size</label>
              <input type="number" value={fontSize}
                onChange={e => setFontSize(parseInt(e.target.value))}
                min={12} max={200}
                className="w-full border rounded-lg p-2 focus:ring-2 focus:ring-blue-500 outline-none" />
            </div>
            <div>
              <label className="block text-sm font-medium mb-1 text-gray-700">Format</label>
              <select value={format} onChange={e => setFormat(e.target.value)}
                className="w-full border rounded-lg p-2 focus:ring-2 focus:ring-blue-500 outline-none">
                <option value="png">PNG</option>
                <option value="jpg">JPG</option>
                <option value="webp">WebP</option>
                <option value="pdf">PDF</option>
              </select>
            </div>
          </div>

          <div className="grid grid-cols-2 gap-6 bg-gray-50 p-4 rounded-lg border">
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-sm font-medium text-gray-700">Text Color</label>
                <label className="inline-flex items-center text-xs cursor-pointer">
                  <input type="checkbox" checked={randomText} onChange={e => setRandomText(e.target.checked)} className="mr-1 rounded" />
                  Random
                </label>
              </div>
              <input 
                type="color" 
                value={color} 
                onChange={e => setColor(e.target.value)}
                disabled={randomText}
                className={`w-full h-10 border rounded-lg cursor-pointer ${randomText ? 'opacity-40 cursor-not-allowed' : ''}`} 
              />
            </div>
            <div>
              <div className="flex justify-between items-center mb-1">
                <label className="text-sm font-medium text-gray-700">Background</label>
                <label className="inline-flex items-center text-xs cursor-pointer">
                  <input type="checkbox" checked={randomBg} onChange={e => setRandomBg(e.target.checked)} className="mr-1 rounded" />
                  Random
                </label>
              </div>
              <input 
                type="color" 
                value={background} 
                onChange={e => setBackground(e.target.value)}
                disabled={randomBg}
                className={`w-full h-10 border rounded-lg cursor-pointer ${randomBg ? 'opacity-40 cursor-not-allowed' : ''}`} 
              />
            </div>
          </div>

          {(randomText || randomBg) && (
            <button 
              type="button"
              onClick={shuffleColors}
              className="w-full border border-gray-300 text-gray-700 py-1.5 rounded-lg hover:bg-gray-50 flex items-center justify-center gap-1.5 font-medium transition"
            >
              🎲 Shuffle Random Colors
            </button>
          )}

          <div>
            <label className="block text-sm font-medium mb-2 text-gray-700">Padding (px)</label>
            <div className="grid grid-cols-4 gap-2">
              <div>
                <label className="block text-xs text-gray-500 mb-1 text-center">Top</label>
                <input type="number" value={paddingTop}
                  onChange={e => setPaddingTop(parseInt(e.target.value))}
                  min={0} max={500}
                  className="w-full border rounded-lg p-2 text-center focus:ring-2 focus:ring-blue-500 outline-none" />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1 text-center">Right</label>
                <input type="number" value={paddingRight}
                  onChange={e => setPaddingRight(parseInt(e.target.value))}
                  min={0} max={500}
                  className="w-full border rounded-lg p-2 text-center focus:ring-2 focus:ring-blue-500 outline-none" />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1 text-center">Bottom</label>
                <input type="number" value={paddingBottom}
                  onChange={e => setPaddingBottom(parseInt(e.target.value))}
                  min={0} max={500}
                  className="w-full border rounded-lg p-2 text-center focus:ring-2 focus:ring-blue-500 outline-none" />
              </div>
              <div>
                <label className="block text-xs text-gray-500 mb-1 text-center">Left</label>
                <input type="number" value={paddingLeft}
                  onChange={e => setPaddingLeft(parseInt(e.target.value))}
                  min={0} max={500}
                  className="w-full border rounded-lg p-2 text-center focus:ring-2 focus:ring-blue-500 outline-none" />
              </div>
            </div>
          </div>

          <button onClick={doRender} disabled={loading}
            className="w-full bg-blue-600 text-white py-2.5 rounded-lg hover:bg-blue-700 font-semibold disabled:opacity-50 transition shadow">
            {loading ? 'Rendering...' : 'Render'}
          </button>
        </div>
        {imageUrl && (
          <div className="mt-6 text-center">
            <img src={imageUrl} alt="Rendered"
              className="mx-auto rounded-lg shadow-md max-w-full border" />
            <a href={imageUrl} download={`khmer-text.${format}`}
              className="mt-3 inline-block text-blue-600 hover:text-blue-800 underline text-sm transition">
              Download Image
            </a>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
