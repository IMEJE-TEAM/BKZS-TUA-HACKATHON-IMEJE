(function () {
  'use strict';

  const BATCH_SIZE   = 5;
  const VIEW_WINDOW  = 800;
  const THRESHOLD    = 0.5;
  const TARGET_FPS   = 30;
  const FRAME_MS     = 1000 / TARGET_FPS;
  const MAX_MINI_LOGS = 40;

  const C = {
    bg:'#c7cddc', surface:'#d6dce9', surface2:'#ced4e2',
    border:'#a5adc2', grid:'#b5bcce', txt:'#3d4a5e',
    heading:'#060b14', green:'#10b981', red:'#ef4444',
    amber:'#f59e0b', purple:'#6366f1', blue:'#3b82f6',
  };

  const FEAT_CFG = [
    { key:'mean_prRes', label:'Ort. Pseudorange Hatasi', color:'#f0883e', unit:'m' },
    { key:'std_prRes',  label:'Std Pseudorange Hatasi',  color:'#ef5350', unit:'m' },
    { key:'max_prRes',  label:'Maks Pseudorange Hatasi', color:'#bc8cff', unit:'m' },
    { key:'mean_cno',   label:'Ort. Sinyal Gucu (C/N0)', color:'#39d2c0', unit:'dBHz' },
    { key:'std_cno',    label:'Std Sinyal Gucu',         color:'#66bb6a', unit:'' },
    { key:'cno_elev_ratio', label:'Sinyal/Yukseklik Orani', color:'#f778ba', unit:'' },
  ];

  let DATA = null;
  let _rawData = null;
  let _dataLoadPromise = null;

  let cursor = 0, paused = false, running = false, speed = BATCH_SIZE;
  let tp = 0, fp = 0, fn = 0, tn = 0;
  let animId = null, lastFrame = 0;
  let charts = [], modelChart = null;
  let miniLogs = [];
  let prevLabel = -1;

  const $ = id => document.getElementById(id);
  const btnStart     = $('btn-start');
  const btnPause     = $('btn-pause');
  const btnSpeedUp   = $('btn-speed-up');
  const btnSpeedDown = $('btn-speed-down');
  const btnReset     = $('btn-reset');
  const speedDisp    = $('speed-display');
  const chartsEl     = $('charts');
  const loadingEl    = $('loading');
  const presimArea   = $('presim-area');
  const simSection   = $('sim-section');
  const simBody      = $('sim-body');
  const simFooter    = $('sim-footer');
  const alarmEl      = $('alarm');
  const navIndicator = $('nav-indicator');
  const heroLogo     = $('hero-logo');

  function rowToTime(idx) {
    const total = DATA ? DATA.totalRows : 30000;
    const sec = Math.floor((idx / total) * 86400);
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    return String(h).padStart(2,'0') + ':' + String(m).padStart(2,'0') + ':' + String(s).padStart(2,'0');
  }

  function initData() {
    DATA = {
      features: FEAT_CFG,
      labels: [],
      probs: [],
      data: {},
      totalRows: 0
    };
    FEAT_CFG.forEach(f => DATA.data[f.key] = []);
  }

  function _preloadData() {
    _dataLoadPromise = fetch('data/simulation_data.json')
      .then(r => r.json())
      .then(d => { _rawData = d; })
      .catch(() => {});
  }

  function loadData() {
    initData();
    buildCharts();
    buildDistBars();
    btnStart.disabled = false;
    _preloadData();
  }

  function buildCharts() {
    chartsEl.innerHTML = '';
    charts = [];
    DATA.features.forEach(f => {
      const panel = document.createElement('div');
      panel.className = 'chart-panel';
      panel.innerHTML = `
        <div class="chart-label">
          <div class="chart-label-left">
            <span class="chart-label-name" style="color:${f.color}">${f.label}</span>
            <span class="chart-label-sub">${f.unit ? '(' + f.unit + ')' : ''}</span>
          </div>
          <span class="chart-label-val" style="color:${f.color};background:${f.color}12" id="val-${f.key}">&#x2014;</span>
        </div>
        <canvas class="chart-canvas" id="cv-${f.key}"></canvas>`;
      chartsEl.appendChild(panel);
      const cv = document.getElementById('cv-' + f.key);
      charts.push({ key: f.key, color: f.color, canvas: cv, ctx: cv.getContext('2d') });
    });
    const mp = document.createElement('div');
    mp.className = 'chart-panel';
    mp.innerHTML = `
      <div class="chart-label">
        <div class="chart-label-left">
          <span class="chart-label-name" style="color:${C.purple}">Saldiri Olasiligi (CRNN)</span>
          <span class="chart-label-sub">Esik: 0.50</span>
        </div>
        <span class="chart-label-val" style="color:${C.purple};background:${C.purple}12" id="val-model">&#x2014;</span>
      </div>
      <canvas class="chart-canvas chart-canvas-model" id="cv-model"></canvas>`;
    chartsEl.appendChild(mp);
    const mcv = document.getElementById('cv-model');
    modelChart = { canvas: mcv, ctx: mcv.getContext('2d') };
  }

  function buildDistBars() {
    const el = $('dist-bars');
    if (!el) return;
    el.innerHTML = '';
    DATA.features.forEach(f => {
      el.innerHTML += `
        <div class="dist-bar-row">
          <span class="dist-bar-label">${f.key.replace('mean_','').replace('std_','\u03c3 ')}</span>
          <div class="dist-bar-track"><div class="dist-bar-fill" id="db-${f.key}" style="background:${f.color};width:0%"></div></div>
          <span class="dist-bar-val" id="dbv-${f.key}">&#x2014;</span>
        </div>`;
    });
  }

  function updateDistBars() {
    if (!DATA || cursor < 10) return;
    DATA.features.forEach(f => {
      const vals = DATA.data[f.key];
      const allMin = Math.min(...vals.slice(0, 500));
      const allMax = Math.max(...vals.slice(0, 500));
      const range = allMax - allMin || 1;
      const cur = vals[cursor - 1];
      const pct = Math.min(100, Math.max(0, ((cur - allMin) / range) * 100));
      const bar = $('db-' + f.key);
      const valEl = $('dbv-' + f.key);
      if (bar) bar.style.width = pct + '%';
      if (valEl) valEl.textContent = cur.toFixed(1);
    });
  }

  function resizeAll() {
    const all = [...charts, modelChart].filter(Boolean);
    all.forEach(c => {
      const dpr = window.devicePixelRatio || 1;
      const rect = c.canvas.getBoundingClientRect();
      c.canvas.width = rect.width * dpr;
      c.canvas.height = rect.height * dpr;
      c.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
      c.w = rect.width;
      c.h = rect.height;
    });
  }

  function drawFeatureChart(ch, vs, ve) {
    const ctx = ch.ctx, w = ch.w, h = ch.h;
    const vals = DATA.data[ch.key];
    const labels = DATA.labels;
    const n = ve - vs;
    if (n < 2 || !w) return;
    ctx.clearRect(0, 0, w, h);
    const sx = w / n;
    for (let i = 0; i < n; i++) {
      if (labels[vs + i] === 1) { ctx.fillStyle = 'rgba(239,68,68,0.06)'; ctx.fillRect(i * sx, 0, sx + 0.5, h); }
    }
    ctx.strokeStyle = C.grid; ctx.lineWidth = 0.5;
    for (let g = 0; g < 4; g++) {
      const gy = h * (g + 1) / 5;
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    }
    let vmin = Infinity, vmax = -Infinity;
    for (let i = vs; i < ve; i++) { const v = vals[i]; if (v < vmin) vmin = v; if (v > vmax) vmax = v; }
    const pad = Math.max((vmax - vmin) * 0.08, 0.5);
    vmin -= pad; vmax += pad;
    const vrange = vmax - vmin || 1;
    const toY = v => h - ((v - vmin) / vrange) * h;
    const toX = i => ((i - vs) / n) * w;
    let prevSpoof = labels[vs] === 1;
    ctx.lineWidth = 1.4; ctx.lineJoin = 'round';
    ctx.beginPath();
    ctx.strokeStyle = prevSpoof ? C.red : ch.color;
    ctx.moveTo(toX(vs), toY(vals[vs]));
    for (let i = vs + 1; i < ve; i++) {
      const isSpoof = labels[i] === 1;
      if (isSpoof !== prevSpoof) {
        ctx.lineTo(toX(i), toY(vals[i])); ctx.stroke();
        ctx.beginPath(); ctx.strokeStyle = isSpoof ? C.red : ch.color; ctx.moveTo(toX(i), toY(vals[i]));
        prevSpoof = isSpoof;
      } else { ctx.lineTo(toX(i), toY(vals[i])); }
    }
    ctx.stroke();
    document.getElementById('val-' + ch.key).textContent = vals[ve - 1].toFixed(2);
  }

  function drawModelChart(vs, ve) {
    if (!modelChart) return;
    const ctx = modelChart.ctx, w = modelChart.w, h = modelChart.h;
    const probs = DATA.probs, labels = DATA.labels;
    const n = ve - vs;
    if (n < 2 || !w) return;
    ctx.clearRect(0, 0, w, h);
    const sx = w / n;
    for (let i = 0; i < n; i++) {
      if (labels[vs + i] === 1) { ctx.fillStyle = 'rgba(239,68,68,0.05)'; ctx.fillRect(i * sx, 0, sx + 0.5, h); }
    }
    ctx.strokeStyle = C.grid; ctx.lineWidth = 0.5;
    for (let g = 0; g < 4; g++) {
      const gy = h * (g + 1) / 5;
      ctx.beginPath(); ctx.moveTo(0, gy); ctx.lineTo(w, gy); ctx.stroke();
    }
    const thY = h - (THRESHOLD * h);
    ctx.strokeStyle = C.amber; ctx.lineWidth = 1; ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(0, thY); ctx.lineTo(w, thY); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.amber; ctx.font = 'bold 10px Inter,sans-serif';
    ctx.fillText('0.50', w - 30, thY - 5);
    const toY = v => h - v * h;
    const toX = i => ((i - vs) / n) * w;
    for (let i = vs; i < ve - 1; i++) {
      const p1 = probs[i], p2 = probs[i + 1];
      const x1 = toX(i), x2 = toX(i + 1), y1 = toY(p1), y2 = toY(p2);
      ctx.fillStyle = (p1 + p2) / 2 >= THRESHOLD ? 'rgba(239,68,68,0.12)' : 'rgba(16,185,129,0.05)';
      ctx.beginPath(); ctx.moveTo(x1, thY); ctx.lineTo(x1, y1); ctx.lineTo(x2, y2); ctx.lineTo(x2, thY);
      ctx.closePath(); ctx.fill();
    }
    ctx.strokeStyle = C.purple; ctx.lineWidth = 4; ctx.globalAlpha = 0.08;
    ctx.beginPath();
    for (let i = vs; i < ve; i++) { const x = toX(i), y = toY(probs[i]); i === vs ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
    ctx.stroke();
    ctx.globalAlpha = 1; ctx.lineWidth = 1.6; ctx.lineJoin = 'round';
    ctx.beginPath();
    for (let i = vs; i < ve; i++) { const x = toX(i), y = toY(probs[i]); i === vs ? ctx.moveTo(x, y) : ctx.lineTo(x, y); }
    ctx.stroke();
    let inAtk = false, atkS = 0;
    for (let i = vs; i <= ve; i++) {
      const above = i < ve && probs[i] >= THRESHOLD;
      if (above && !inAtk) { atkS = i; inAtk = true; }
      else if (!above && inAtk) {
        if (i - atkS > 10) {
          ctx.fillStyle = 'rgba(239,68,68,.45)'; ctx.font = 'bold 9px Inter,sans-serif'; ctx.textAlign = 'center';
          ctx.fillText('SALDIRI', toX(atkS + (i - atkS) / 2), 14); ctx.textAlign = 'start';
        }
        inAtk = false;
      }
    }
    document.getElementById('val-model').textContent = probs[ve - 1].toFixed(3);
  }

  function addMiniLog(idx) {
    const label = DATA.labels[idx];
    const prob = DATA.probs[idx];
    const time = rowToTime(idx);
    if (label !== prevLabel || miniLogs.length === 0) {
      miniLogs.unshift({ time, type: label === 1 ? 'spoof' : 'normal', prob, idx });
      if (miniLogs.length > MAX_MINI_LOGS) miniLogs.pop();
      prevLabel = label;
    }
  }

  function renderMiniLogs() {
    const feed = $('mini-log-feed');
    if (!feed) return;
    let html = '';
    const shown = miniLogs.slice(0, 15);
    for (const e of shown) {
      const cls = e.type === 'spoof' ? 'spoof' : 'normal';
      const typeLabel = e.type === 'spoof' ? 'SPF' : 'NRM';
      html += `<div class="mini-log-entry ${cls}"><span class="mini-log-time">${e.time}</span><span class="mini-log-type">${typeLabel}</span><span class="mini-log-prob">${e.prob.toFixed(3)}</span></div>`;
    }
    feed.innerHTML = html;
  }

  function syncState() {
    try {
      localStorage.setItem('bkzs_sim', JSON.stringify({ cursor, tp, fp, fn, tn, running, paused, speed, timestamp: Date.now() }));
    } catch(e) {}
    if (window._bkzsSim) {
      window._bkzsSim.running = running;
      window._bkzsSim.paused = paused;
      window._bkzsSim.speed = speed;
      window._bkzsSim.cursor = cursor;
    }
  }

  function updateDashboard() {
    const total = DATA.totalRows;
    const pct = total > 0 ? Math.min(100, cursor / total * 100) : 0;
    $('progress-pct').textContent = pct.toFixed(0) + '%';
    $('progress-fill').style.width = pct + '%';
    $('progress-detail').textContent = cursor.toLocaleString() + ' / ' + total.toLocaleString();
    $('s-processed').textContent = cursor.toLocaleString();
    $('s-processed-bar').style.width = pct + '%';
    const prec = tp / Math.max(tp + fp, 1);
    const rec  = tp / Math.max(tp + fn, 1);
    const f1   = 2 * prec * rec / Math.max(prec + rec, 1e-9);
    $('s-precision').textContent = prec.toFixed(3);
    $('s-recall').textContent = rec.toFixed(3);
    $('s-f1').textContent = f1.toFixed(3);
    $('s-precision-bar').style.width = (prec * 100) + '%';
    $('s-recall-bar').style.width = (rec * 100) + '%';
    $('s-f1-bar').style.width = (f1 * 100) + '%';
    $('cm-tp').textContent = tp.toLocaleString();
    $('cm-fp').textContent = fp.toLocaleString();
    $('cm-fn').textContent = fn.toLocaleString();
    $('cm-tn').textContent = tn.toLocaleString();
    const curProb = cursor > 0 ? DATA.probs[cursor - 1] : 0;
    if (curProb >= THRESHOLD) {
      alarmEl.className = 'alarm-on';
      alarmEl.innerHTML = '<div class="alarm-text"><span>\u26a0</span> SPOOFING SALDIRISI TESPIT EDILDI</div>';
    } else {
      alarmEl.className = '';
      alarmEl.innerHTML = '';
    }
    if (running) {
      const dot = navIndicator.querySelector('.indicator-dot');
      const txt = navIndicator.querySelector('span:last-child');
      if (paused) {
        dot.style.background = C.amber;
        dot.style.boxShadow = '0 0 8px rgba(245,158,11,.4)';
        txt.textContent = 'Duraklatildi';
      } else {
        dot.style.background = C.green;
        dot.style.boxShadow = '0 0 8px rgba(16,185,129,.4)';
        txt.textContent = 'Calisiyor';
      }
    }
    const statusEl = $('sim-status-text');
    if (statusEl) {
      const det = (tp + fn) > 0 ? (tp / (tp + fn) * 100).toFixed(1) : '0.0';
      statusEl.textContent = cursor.toLocaleString() + ' satir | F1: ' + f1.toFixed(3) + ' | Tespit: %' + det;
    }
  }

  function step() {
    if (!paused && cursor < DATA.totalRows) {
      const nc = Math.min(cursor + speed, DATA.totalRows);
      for (let i = cursor; i < nc; i++) {
        const pred = DATA.probs[i] >= THRESHOLD ? 1 : 0;
        const real = DATA.labels[i];
        if (pred === 1 && real === 1) tp++;
        else if (pred === 1 && real === 0) fp++;
        else if (pred === 0 && real === 1) fn++;
        else tn++;
        addMiniLog(i);
      }
      cursor = nc;
    }
    const ve = cursor, vs = Math.max(0, ve - VIEW_WINDOW);
    if (ve > 1) {
      charts.forEach(ch => drawFeatureChart(ch, vs, ve));
      drawModelChart(vs, ve);
    }
    updateDashboard();
    updateDistBars();
    renderMiniLogs();
    syncState();
  }

  function animate(ts) {
    if (!running) return;
    if (ts - lastFrame >= FRAME_MS) { lastFrame = ts; step(); }
    animId = requestAnimationFrame(animate);
  }

  function revealSimArea() {
    if (presimArea && presimArea.style.display !== 'none') {
      if (heroLogo) heroLogo.classList.add('logo-fly-out');
      presimArea.classList.add('hero-fade-out');
      setTimeout(() => {
        presimArea.style.display = 'none';
        presimArea.classList.remove('hero-fade-out');
        simSection.style.display = '';
        loadingEl.style.display = 'none';
        simBody.style.display = '';
        simFooter.style.display = '';
        btnPause.textContent = '\u23f8 Duraklat';
        requestAnimationFrame(() => { resizeAll(); });
        lastFrame = 0;
        if (animId) cancelAnimationFrame(animId);
        animId = requestAnimationFrame(animate);
      }, 900);
    } else {
      loadingEl.style.display = 'none';
      simBody.style.display = '';
      simFooter.style.display = '';
      btnPause.textContent = '\u23f8 Duraklat';
      requestAnimationFrame(() => { resizeAll(); });
      lastFrame = 0;
      if (animId) cancelAnimationFrame(animId);
      animId = requestAnimationFrame(animate);
    }
  }

  async function startSimulation() {
    cursor = 0; tp = fp = fn = tn = 0;
    paused = false; running = true;
    miniLogs = []; prevLabel = -1;
    buildCharts();
    buildDistBars();
    btnStart.disabled = true;
    if (!_rawData) {
      btnStart.textContent = '\u23f3 Hazirlanıyor...';
      try {
        if (_dataLoadPromise) await _dataLoadPromise;
        if (!_rawData) {
          const res = await fetch('data/simulation_data.json');
          _rawData = await res.json();
        }
      } catch(e) {
        btnStart.disabled = false;
        btnStart.textContent = '\u25b6 Simulasyonu Baslat';
        running = false;
        return;
      }
    }
    DATA = _rawData;
    window._bkzsSim = { data: DATA, running: true, paused: false, speed, cursor: 0 };
    revealSimArea();
  }

  function resetSimulation() {
    cursor = 0; tp = fp = fn = tn = 0;
    paused = false; running = false;
    miniLogs = []; prevLabel = -1;
    if (animId) { cancelAnimationFrame(animId); animId = null; }
    if (window._bkzsSim) window._bkzsSim.running = false;
    initData();
    buildCharts();
    buildDistBars();
    updateDashboard();
    if (presimArea) { presimArea.style.display = ''; if (heroLogo) heroLogo.classList.remove('logo-fly-out'); }
    if (simSection) simSection.style.display = 'none';
    btnStart.disabled = false;
    btnStart.textContent = '\u25b6 Simulasyonu Baslat';
    const connEl = $('connection-status');
    if (connEl) { connEl.textContent = ''; connEl.className = 'conn-badge'; }
    const dot = navIndicator.querySelector('.indicator-dot');
    const txt = navIndicator.querySelector('span:last-child');
    if (dot) { dot.style.background = ''; dot.style.boxShadow = ''; }
    if (txt) txt.textContent = 'Hazir';
  }

  btnStart.addEventListener('click', startSimulation);
  btnPause.addEventListener('click', () => {
    paused = !paused;
    btnPause.textContent = paused ? '\u25b6 Devam' : '\u23f8 Duraklat';
    syncState();
  });
  btnSpeedUp.addEventListener('click', () => { speed = Math.min(speed + 5, 100); speedDisp.textContent = speed + 'x'; syncState(); });
  btnSpeedDown.addEventListener('click', () => { speed = Math.max(speed - 5, 1); speedDisp.textContent = speed + 'x'; syncState(); });
  btnReset.addEventListener('click', resetSimulation);

  document.addEventListener('keydown', e => {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.code === 'Space') { e.preventDefault(); if (running) { paused = !paused; btnPause.textContent = paused ? '\u25b6 Devam' : '\u23f8 Duraklat'; syncState(); } }
    if (e.code === 'KeyR' && running) { e.preventDefault(); resetSimulation(); }
  });

  window.addEventListener('resize', () => { if (running) resizeAll(); });

  window.showPage = function(name, linkId) {
    document.querySelectorAll('.page-wrap').forEach(p => p.classList.remove('active'));
    const page = document.getElementById('page-' + name);
    if (page) page.classList.add('active');
    document.querySelectorAll('.nav-link').forEach(l => l.classList.remove('active'));
    const linkEl = (typeof linkId === 'string') ? document.getElementById(linkId) : linkId;
    if (linkEl) linkEl.classList.add('active');
    if (name === 'model' && window._onModelPageShow) window._onModelPageShow();
  };

  loadData();
})();
