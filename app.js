 // ==== Estado global ====
    const els = {
      status: document.getElementById('status'),
      dropzone: document.getElementById('dropzone'),
      fileInput: document.getElementById('fileInput'),
      analyzeBtn: document.getElementById('analyzeBtn'),
      clearBtn: document.getElementById('clearBtn'),
      canvas: document.getElementById('canvas'),
      legend: document.getElementById('legend'),
      clsResults: document.getElementById('clsResults'),
      detResults: document.getElementById('detResults'),
      comment: document.getElementById('comment'),
      saveComment: document.getElementById('saveComment'),
      sentBar: document.getElementById('sentBar'),
      sentScore: document.getElementById('sentScore'),
      emotionChips: document.getElementById('emotionChips'),
      history: document.getElementById('history'),
      exportBtn: document.getElementById('exportBtn'),
    };

    const ctx = els.canvas.getContext('2d');
    let currentImage = null; // HTMLImageElement
    let mobilenetModel, cocoModel, useModel;
    let anchorEmbeddings = null; // { posE, negE, emoE }
    let lastDetections = [], lastClassification = [];

    // ==== Utilidades ====
    const sleep = (ms) => new Promise(r => setTimeout(r, ms));
    function setStatus(msg){ els.status.textContent = msg || ''; }
    function toDataURL(img) {
      const c = document.createElement('canvas');
      const maxW = 320;
      const scale = Math.min(1, maxW / img.width);
      c.width = Math.round(img.width * scale);
      c.height = Math.round(img.height * scale);
      const cctx = c.getContext('2d');
      cctx.drawImage(img, 0, 0, c.width, c.height);
      return c.toDataURL('image/jpeg', .85);
    }
    function download(filename, data){
      const blob = new Blob([data], {type: 'application/json'});
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url; a.download = filename; document.body.appendChild(a); a.click(); a.remove();
      URL.revokeObjectURL(url);
    }

    // Coseno para arrays (más rápido y estable que leer tensors constantemente)
    function cosineArray(a, b){
      const al = a.length;
      let dot = 0, na = 0, nb = 0;
      for(let i=0;i<al;i++){ const av=a[i], bv=b[i]; dot += av*bv; na += av*av; nb += bv*bv; }
      return dot / (Math.sqrt(na) * Math.sqrt(nb) + 1e-8);
    }
    function scale01(x){ return Math.max(0, Math.min(1, (x+1)/2)); }

    // ==== Carga de modelos ====
    async function loadModels(){
      setStatus('Cargando modelos…');
      [mobilenetModel, cocoModel, useModel] = await Promise.all([
        mobilenet.load({version:2, alpha:1.0}),
        cocoSsd.load({base:'lite_mobilenet_v2'}),
        use.load(),
      ]);

      // Precalcular anclas semánticas para sentimiento (las convertimos a arrays para cálculos rápidos)
      const labels = {
        positive: ['positive', 'good', 'great', 'love', 'excellent', 'awesome', 'happy'],
        negative: ['negative', 'bad', 'terrible', 'hate', 'awful', 'angry', 'sad'],
        emotions: {
          happy: ['happy','joyful','pleased','content'],
          angry: ['angry','furious','irritated','annoyed'],
          sad: ['sad','down','unhappy','depressed'],
          fear: ['afraid','scared','fearful','anxious'],
          surprise: ['surprised','amazed','astonished','shocked'],
          disgust: ['disgusted','gross','revolted','nauseated'],
          neutral: ['neutral','calm','okay','fine']
        }
      };

      const all = [labels.positive, labels.negative, ...Object.values(labels.emotions)];
      const flat = all.flat();

      // Obtenemos el embedding para todas las frases en un solo paso y lo convertimos a arrays
      const embs = await useModel.embed(flat);
      const embsArr = await embs.array(); // [[...], [...], ...]
      embs.dispose();

      let idx = 0;
      const take = (arr) => {
        const out = arr.map((_, i) => Float32Array.from(embsArr[idx + i]));
        idx += arr.length;
        return out;
      };

      const posE = take(labels.positive);
      const negE = take(labels.negative);
      const emoE = Object.fromEntries(Object.entries(labels.emotions).map(([k, v]) => [k, take(v)]));

      anchorEmbeddings = { posE, negE, emoE };

      setStatus('Modelos listos. Sube una imagen.');
      els.analyzeBtn.disabled = !currentImage;
    }

    // ==== Imagen y dibujo ====
    function drawImageToCanvas(img){
      const maxW = 1280, maxH = 820;
      let {width:w, height:h} = img;
      const s = Math.min(1, Math.min(maxW/w, maxH/h));
      const W = Math.round(w*s), H = Math.round(h*s);
      els.canvas.width = W; els.canvas.height = H;
      ctx.clearRect(0,0,W,H);
      ctx.drawImage(img, 0, 0, W, H);
    }

    function drawDetections(dets){
      const W = els.canvas.width, H = els.canvas.height;
      ctx.lineWidth = 2; ctx.font = '14px ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto';
      dets.forEach((d, i)=>{
        const [x,y,w,h] = d.bbox;
        const color = `hsl(${(i*53)%360} 80% 60%)`;
        ctx.strokeStyle = color; ctx.fillStyle = color; ctx.globalAlpha = .85; ctx.strokeRect(x,y,w,h);
        const label = `${d.class} ${(d.score*100).toFixed(1)}%`;
        const tw = ctx.measureText(label).width + 8; const th = 18;
        ctx.fillRect(x, Math.max(0, y-th), tw, th);
        ctx.fillStyle = '#081022'; ctx.globalAlpha = 1; ctx.fillText(label, x+4, Math.max(12, y-4));
      });
    }

    function updateLegend(colors){
      els.legend.innerHTML = '';
      if(!colors || !colors.length){ els.legend.setAttribute('aria-hidden','true'); return; }
      els.legend.removeAttribute('aria-hidden');
      colors.forEach(c=>{
        const span = document.createElement('span'); span.className = 'pill'; span.style.borderColor = c; span.textContent = '■'; span.style.color = c; els.legend.appendChild(span);
      });
    }

    // ==== Clasificación y Detección ====
    async function analyze(){
      if(!currentImage || !mobilenetModel || !cocoModel){ return; }
      setStatus('Analizando imagen…');
      drawImageToCanvas(currentImage);

      // Clasificación
      const cls = await mobilenetModel.classify(currentImage, 5);
      lastClassification = cls;
      els.clsResults.innerHTML = '';
      cls.forEach(item=>{
        const row = document.createElement('div'); row.className = 'row'; row.setAttribute('role','listitem');
        const left = document.createElement('div'); left.innerHTML = `<span class="label">${item.className}</span>`;
        const right = document.createElement('div'); right.textContent = `${(item.probability*100).toFixed(2)}%`;
        row.append(left, right); els.clsResults.appendChild(row);
      });

      // Detección de objetos
      const detsRaw = await cocoModel.detect(currentImage, 20);
      // Convertir bboxes a escala del canvas
      const scaleX = els.canvas.width / currentImage.width;
      const scaleY = els.canvas.height / currentImage.height;
      const dets = detsRaw.map(d=>({
        ...d,
        bbox: [d.bbox[0]*scaleX, d.bbox[1]*scaleY, d.bbox[2]*scaleX, d.bbox[3]*scaleY]
      }));
      lastDetections = dets;

      // Dibujar
      drawDetections(dets);
      updateLegend(dets.map((_,i)=>`hsl(${(i*53)%360} 80% 60%)`));

      // Mostrar lista
      els.detResults.innerHTML = '';
      if(dets.length===0){
        const empty = document.createElement('div'); empty.className='muted'; empty.textContent = 'No se detectaron objetos.'; els.detResults.appendChild(empty);
      } else {
        dets.forEach((d,i)=>{
          const row = document.createElement('div'); row.className='row'; row.setAttribute('role','listitem');
          row.innerHTML = `<div class="label">${i+1}. ${d.class}</div><div>${(d.score*100).toFixed(1)}%</div>`;
          els.detResults.appendChild(row);
        });
      }

      setStatus('Análisis completado. Puedes comentar y guardar en el historial.');
      els.saveComment.disabled = false;
    }

    // ==== Sentimiento (USE + anclas) ====
    async function estimateSentiment(text){
      if(!text || !useModel || !anchorEmbeddings) return {score:0, emotions:[], raw:{} };

      // Obtener embedding como array para evitar llamadas sincronas innecesarias
      const embTensor = await useModel.embed([text]);
      const embArrs = await embTensor.array();
      embTensor.dispose();
      const embArr = Float32Array.from(embArrs[0]);

      const meanSim = (arrOfEmb) => arrOfEmb.map(e => cosineArray(embArr, e)).reduce((a,b)=>a+b,0)/arrOfEmb.length;
      const pos = meanSim(anchorEmbeddings.posE);
      const neg = meanSim(anchorEmbeddings.negE);
      const raw = {pos, neg};
      const score = Math.max(-1, Math.min(1, pos - neg));

      const emoScores = Object.entries(anchorEmbeddings.emoE).map(([k,arr])=>[k, meanSim(arr)]);
      emoScores.sort((a,b)=>b[1]-a[1]);
      const emotions = emoScores.slice(0,3).map(([k,v])=>({label:k, score: v}));

      return {score, emotions, raw};
    }

    function renderSentimentUI(score, emotions){
      const pct = scale01(score)*100; // 0..100
      els.sentBar.style.width = pct + '%';
      els.sentBar.style.background = score>=0 ? `linear-gradient(90deg, var(--accent-2), var(--ok))` : `linear-gradient(90deg, var(--danger), var(--warn))`;
      els.sentScore.textContent = score.toFixed(2);
      els.emotionChips.innerHTML = '';
      emotions.forEach(e=>{
        const chip = document.createElement('span'); chip.className='chip'; chip.textContent = `${e.label} ${(e.score*100).toFixed(0)}%`;
        els.emotionChips.appendChild(chip);
      });
    }

    // ==== Historial (localStorage) ====
    const LS_KEY = 'image-analyzer-history-v1';
    function loadHistory(){
      try{ return JSON.parse(localStorage.getItem(LS_KEY)||'[]'); }catch{ return [] }
    }
    function saveHistory(items){ localStorage.setItem(LS_KEY, JSON.stringify(items)); }
    function addToHistory(entry){
      const items = loadHistory(); items.unshift(entry); saveHistory(items); renderHistory();
    }
    function removeFromHistory(id){
      const items = loadHistory().filter(x=>x.id!==id); saveHistory(items); renderHistory();
    }
    function renderHistory(){
      const items = loadHistory(); els.history.innerHTML='';
      if(items.length===0){
        const empty = document.createElement('div'); empty.className='muted'; empty.textContent='Aún no hay historial.'; els.history.appendChild(empty); return;
      }
      items.forEach(item=>{
        const div = document.createElement('div'); div.className='item'; div.setAttribute('role','listitem');
        const time = new Date(item.ts).toLocaleString();
        div.innerHTML = `
          <div style="display:flex; justify-content:space-between; gap:.5rem; align-items:center">
            <strong>${time}</strong>
            <button type="button" aria-label="Eliminar del historial" title="Eliminar" style="background:#1a2340;border:1px solid #2a3961;color:#e0e8ff;padding:.3rem .5rem;border-radius:8px;cursor:pointer" data-del="${item.id}">Eliminar</button>
          </div>
          <img src="${item.thumb}" alt="Miniatura del análisis" loading="lazy"/>
          <div class="muted">${item.comment ? 'Comentario: '+item.comment : 'Sin comentario'}</div>
          <div class="muted">Sentimiento: ${(item.sentiment.score>=0?'positivo':'negativo')} (${item.sentiment.score.toFixed(2)})</div>
        `;
        els.history.appendChild(div);
      });
      // Wire eliminar
      els.history.querySelectorAll('[data-del]').forEach(btn=>{
        btn.addEventListener('click', ()=> removeFromHistory(btn.getAttribute('data-del')));
      });
    }

    // ==== Entrada / UX ====
    function handleFiles(files){
      const f = files && files[0]; if(!f) return;
      if(!f.type.startsWith('image/')){ setStatus('Por favor selecciona un archivo de imagen.'); return; }
      const img = new Image(); img.alt = f.name; img.onload = ()=>{
        currentImage = img; drawImageToCanvas(img); els.analyzeBtn.disabled = false; setStatus('Imagen cargada. Pulsa Analizar.');
      };
      const reader = new FileReader(); reader.onload = e=> img.src = e.target.result; reader.readAsDataURL(f);
    }

    // Drag&Drop accesible
    els.dropzone.addEventListener('click', ()=> els.fileInput.click());
    els.dropzone.addEventListener('keydown', (e)=>{ if(e.key==='Enter' || e.key===' '){ e.preventDefault(); els.fileInput.click(); }});
    ['dragenter','dragover'].forEach(ev=> els.dropzone.addEventListener(ev, e=>{ e.preventDefault(); e.stopPropagation(); els.dropzone.classList.add('dragover'); }));
    ['dragleave','drop'].forEach(ev=> els.dropzone.addEventListener(ev, e=>{ e.preventDefault(); e.stopPropagation(); els.dropzone.classList.remove('dragover'); }));
    els.dropzone.addEventListener('drop', (e)=>{ handleFiles(e.dataTransfer.files); });
    els.fileInput.addEventListener('change', (e)=> handleFiles(e.target.files));

    // Botones
    els.analyzeBtn.addEventListener('click', analyze);
    els.clearBtn.addEventListener('click', ()=>{
      currentImage = null; lastDetections=[]; lastClassification=[];
      ctx.clearRect(0,0,els.canvas.width, els.canvas.height);
      els.clsResults.innerHTML = '<div class="muted">Aún sin resultados. Sube una imagen para comenzar.</div>';
      els.detResults.innerHTML = '';
      els.legend.innerHTML=''; els.legend.setAttribute('aria-hidden','true');
      els.analyzeBtn.disabled = true; els.saveComment.disabled = true; setStatus('');
      els.fileInput.value='';
    });

    // Comentario + sentimiento tiempo real
    let sentiLock = 0;
    els.comment.addEventListener('input', async ()=>{
      const txt = els.comment.value.trim();
      els.saveComment.disabled = (txt.length===0 || !currentImage);
      if(!txt){ renderSentimentUI(0, []); return; }
      const my = ++sentiLock; // evitar condiciones de carrera
      const result = await estimateSentiment(txt);
      if(my!==sentiLock) return; // llegó una respuesta vieja
      renderSentimentUI(result.score, result.emotions);
    });

    // Guardar comentario al historial
    els.saveComment.addEventListener('click', async ()=>{
      const txt = els.comment.value.trim();
      const senti = await estimateSentiment(txt||'');
      const entry = {
        id: String(Date.now()),
        ts: Date.now(),
        thumb: currentImage ? toDataURL(currentImage) : '',
        classification: lastClassification,
        detections: lastDetections,
        comment: txt,
        sentiment: senti,
      };
      addToHistory(entry);
      setStatus('Guardado en historial.');

      els.comment.value=''; els.comment.focus();
    });

    // Exportar JSON
    els.exportBtn.addEventListener('click', ()=>{
      const items = loadHistory();
      const pretty = JSON.stringify(items, null, 2);
      download('analisis_imagenes.json', pretty);
    });

    // Accesos rápidos
    window.addEventListener('keydown', (e)=>{
      if(e.altKey && (e.key==='a' || e.key==='A')){ if(!els.analyzeBtn.disabled){ e.preventDefault(); analyze(); } }
      if(e.altKey && (e.key==='s' || e.key==='S')){ if(!els.saveComment.disabled){ e.preventDefault(); els.saveComment.click(); } }
    });

    // Init
    renderHistory();
    loadModels();