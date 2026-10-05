"""Browser checks for pixel-art methods and local palette controls."""
import smoke_playback as harness


def checks(page):
    page.call('Runtime.enable')
    page.wait_for('!!window.privacyDemo', bool)
    result = page.evaluate("""(async () => {
      const {Pixelator}=await import('./js/privacy.js');
      const select=document.querySelector('#style-select');
      select.value='retro'; select.dispatchEvent(new Event('change'));
      if(document.querySelector('#pixel-art-controls').hidden) throw Error('Controls hidden');
      const preset=document.querySelector('#art-preset');
      preset.value='mono'; preset.dispatchEvent(new Event('change'));
      const methods=['none','bayer2','bayer4','bayer8','ordered','clustered','floyd','atkinson'];
      const hashes=[];
      for(const method of methods) {
        const d=document.querySelector('#art-dither'); d.value=method; d.dispatchEvent(new Event('change'));
        const strength=document.querySelector('#art-strength'); strength.value='100'; strength.dispatchEvent(new Event('input'));
        const pixelator=new Pixelator(privacyDemo.CONFIG.privacy);
        const data=new Uint8ClampedArray(32*32*4);
        for(let y=0;y<32;y++) for(let x=0;x<32;x++) {
          const i=(y*32+x)*4; data[i]=data[i+1]=data[i+2]=x*8+y%7; data[i+3]=255;
        }
        pixelator._quantize(data,32,[{ax:0,ay:0,w:32,h:32}]);
        let h=0,whites=0;
        for(let i=0;i<data.length;i+=4) {
          if(![0,255].includes(data[i]) || data[i]!==data[i+1] || data[i]!==data[i+2]) throw Error('Invalid palette color '+method);
          if(data[i]===255) whites++;
          h=Math.imul(h ^ data[i],16777619);
        }
        if(!whites || whites===1024) throw Error('Degenerate output '+method);
        hashes.push(h);
      }
      if(new Set(hashes).size<4) throw Error('Dithering methods do not differ');
      const palette=document.querySelector('#art-palette');
      palette.value='#112233 #ddeeff'; document.querySelector('#art-apply').click();
      document.querySelector('#art-save').click();
      preset.value='green'; preset.dispatchEvent(new Event('change'));
      document.querySelector('#art-load').click();
      if(privacyDemo.CONFIG.privacy.palette.join(' ')!=='#112233 #ddeeff') throw Error('Local palette roundtrip');
      palette.value='invalid'; document.querySelector('#art-apply').click();
      if(privacyDemo.CONFIG.privacy.palette[0]!=='#112233') throw Error('Invalid palette replaced valid state');
      const input=document.querySelector('#art-import');
      const transfer=new DataTransfer();
      transfer.items.add(new File([JSON.stringify({palette:['#ff0000','#00ff00','#0000ff']})], 'palette.json', {type:'application/json'}));
      input.files=transfer.files; input.dispatchEvent(new Event('change'));
      await new Promise(r=>setTimeout(r,100));
      if(privacyDemo.CONFIG.privacy.palette.length!==3) throw Error('JSON import failed');
      const createURL=URL.createObjectURL, anchorClick=HTMLAnchorElement.prototype.click;
      let exported;
      try {
        URL.createObjectURL=blob=>{exported=blob; return createURL(blob);};
        HTMLAnchorElement.prototype.click=function(){};
        document.querySelector('#art-export').click();
      } finally {URL.createObjectURL=createURL; HTMLAnchorElement.prototype.click=anchorClick;}
      if(JSON.parse(await exported.text()).palette[0]!=='#ff0000') throw Error('JSON export failed');
      document.querySelector('#tuning-reset').click();
      if(privacyDemo.CONFIG.privacy.palette.length!==8 || privacyDemo.CONFIG.privacy.ditherStrength!==20) throw Error('Reset failed');
      return {methods:methods.length,variants:new Set(hashes).size,uploads:privacyDemo.getNetworkReport().imageUploads};
    })()""")
    assert result['uploads'] == 0, result
    assert not page.errors, page.errors
    print('PASS: dithering, palette colors, local save/load, invalid input, JSON import, reset:', result)


if __name__ == '__main__':
    harness.checks = checks
    harness.main()
