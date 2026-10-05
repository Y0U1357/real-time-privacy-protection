// Local palette/dithering controls inspired by Image-to-Pixel's editor.
// No remote palettes, CDN, image upload or model/session changes.
export function bindPixelArtControls(settings, repaint) {
  const byId = id => document.getElementById(id);
  const panel = byId('pixel-art-controls'), dither = byId('art-dither');
  const strength = byId('art-strength'), value = byId('art-strength-value');
  const palette = byId('art-palette'), preset = byId('art-preset');
  const message = byId('art-message'), swatches = byId('art-swatches');
  const initial = {palette:[...settings.palette], dither:settings.dither, ditherStrength:settings.ditherStrength};
  const presets = {default:initial.palette, mono:['#000000','#ffffff'], green:['#0f380f','#306230','#8bac0f','#9bbc0f']};
  function parse(input) {
    const colors = Array.isArray(input) ? input : String(input).trim().split(/[\s,;]+/);
    if (colors.length < 2 || colors.length > 16 || colors.some(c => typeof c !== 'string' || !/^#[0-9a-f]{6}$/i.test(c))) {
      throw new Error('Enter 2-16 colors in #RRGGBB format.');
    }
    return colors.map(c => c.toLowerCase());
  }
  function sync() {
    panel.hidden = settings.style !== 'retro';
    dither.value = settings.dither;
    strength.value = settings.ditherStrength;
    value.textContent = settings.ditherStrength + '%';
    palette.value = settings.palette.join(' ');
    swatches.replaceChildren(...settings.palette.map(color => {
      const chip = document.createElement('span');
      chip.style.backgroundColor = color;
      chip.title = color;
      chip.setAttribute('aria-label', color);
      return chip;
    }));
  }
  function apply(colors) {
    try {
      settings.palette = parse(colors);
      message.textContent = 'Palette applied to person regions.';
      sync(); repaint();
    } catch (error) { message.textContent = error.message; }
  }
  dither.addEventListener('change', () => {settings.dither=dither.value; repaint();});
  strength.addEventListener('input', () => {
    settings.ditherStrength=Number(strength.value);
    value.textContent=settings.ditherStrength+'%'; repaint();
  });
  preset.addEventListener('change', () => {if (presets[preset.value]) apply(presets[preset.value]);});
  byId('art-apply').addEventListener('click', () => {preset.value='custom'; apply(palette.value);});
  byId('art-save').addEventListener('click', () => {
    try {localStorage.setItem('privacy.pixelPalette',JSON.stringify(parse(palette.value))); message.textContent='Palette saved on this device.';}
    catch(error) {message.textContent=error.message;}
  });
  byId('art-load').addEventListener('click', () => {
    try {
      const saved=localStorage.getItem('privacy.pixelPalette');
      if (!saved) throw new Error('No saved palette on this device.');
      preset.value='custom'; apply(JSON.parse(saved));
    } catch(error) {message.textContent=error.message;}
  });
  byId('art-export').addEventListener('click', () => {
    const url=URL.createObjectURL(new Blob([JSON.stringify({palette:settings.palette},null,2)],{type:'application/json'}));
    const link=document.createElement('a'); link.href=url; link.download='pixel-palette.json';
    link.click(); setTimeout(()=>URL.revokeObjectURL(url),1000);
  });
  byId('art-import').addEventListener('change', async event => {
    try {
      const file=event.target.files[0]; if (!file) return;
      if (file.size>65536) throw new Error('Palette JSON must be smaller than 64 KB.');
      const json=JSON.parse(await file.text());
      preset.value='custom'; apply(Array.isArray(json) ? json : json.palette);
    } catch(error) {message.textContent='Cannot import palette: '+error.message;}
    finally {event.target.value='';}
  });
  byId('style-select').addEventListener('change', sync);
  byId('tuning-reset').addEventListener('click', () => {
    Object.assign(settings,initial,{palette:[...initial.palette]}); preset.value='default'; sync(); repaint();
  });
  sync();
}
