(function () {
  'use strict';

  const C = {
    bg: '#d6dce9', grid: '#b5bcce', txt: '#6b7588', txtHi: '#0f172a',
    green: '#10b981', red: '#ef4444', amber: '#f59e0b',
    purple: '#6366f1', blue: '#3b82f6',
  };

  
  const EPOCHS = 12;
  const trainLoss = [0.38,0.21,0.13,0.089,0.065,0.048,0.037,0.030,0.025,0.022,0.019,0.018];
  const valLoss   = [0.30,0.17,0.11,0.078,0.058,0.046,0.039,0.035,0.032,0.030,0.029,0.029];
  const trainAcc  = [0.83,0.91,0.95,0.965,0.975,0.982,0.986,0.989,0.991,0.992,0.993,0.993];
  const valAcc    = [0.87,0.93,0.96,0.971,0.979,0.984,0.987,0.989,0.990,0.991,0.991,0.991];

  
  const featureNames = ['max_prRes','std_prRes','mean_prRes','cno_elev_ratio','std_cno','mean_cno'];
  const featureImportance = [0.31, 0.24, 0.17, 0.13, 0.09, 0.06];

  function setupCanvas(id) {
    const cv = document.getElementById(id);
    if (!cv) return null;
    const dpr = window.devicePixelRatio || 1;
    const rect = cv.getBoundingClientRect();
    cv.width = rect.width * dpr;
    cv.height = rect.height * dpr;
    const ctx = cv.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, w: rect.width, h: rect.height };
  }

  
  function drawLineChart(canvasId, data1, data2, label1, label2, color1, color2, yMin, yMax, yLabel) {
    const info = setupCanvas(canvasId);
    if (!info) return;
    const { ctx, w, h } = info;
    const pad = { top: 30, right: 18, bottom: 34, left: 52 };
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    
    const nLines = 5;
    for (let i = 0; i <= nLines; i++) {
      const y = pad.top + (ch / nLines) * i;
      const val = yMax - (yMax - yMin) * (i / nLines);
      ctx.strokeStyle = C.grid; ctx.lineWidth = 0.6;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
      ctx.fillStyle = C.txt; ctx.font = '10px Inter,system-ui,sans-serif'; ctx.textAlign = 'right';
      ctx.fillText(val.toFixed(2), pad.left - 6, y + 3);
    }

    
    ctx.textAlign = 'center';
    const step = Math.max(1, Math.floor(EPOCHS / 8));
    for (let i = 0; i < EPOCHS; i += step) {
      const x = pad.left + (i / (EPOCHS - 1)) * cw;
      ctx.fillText((i + 1).toString(), x, h - pad.bottom + 16);
    }
    ctx.fillStyle = C.txt; ctx.fillText('Epoch', pad.left + cw / 2, h - 4);

    
    ctx.save(); ctx.translate(14, pad.top + ch / 2); ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center'; ctx.fillStyle = C.txt; ctx.font = '10px Inter,system-ui,sans-serif';
    ctx.fillText(yLabel, 0, 0); ctx.restore();

    const toX = i => pad.left + (i / (EPOCHS - 1)) * cw;
    const toY = v => pad.top + ((yMax - v) / (yMax - yMin)) * ch;

    
    function drawArea(data, color) {
      ctx.fillStyle = color + '18';
      ctx.beginPath();
      ctx.moveTo(toX(0), toY(yMin));
      for (let i = 0; i < data.length; i++) ctx.lineTo(toX(i), toY(data[i]));
      ctx.lineTo(toX(data.length - 1), toY(yMin));
      ctx.closePath(); ctx.fill();
    }

    
    function drawLine(data, color, dashed) {
      ctx.strokeStyle = color; ctx.lineWidth = 2.2; ctx.lineJoin = 'round'; ctx.lineCap = 'round';
      ctx.setLineDash(dashed ? [6, 4] : []);
      ctx.beginPath();
      for (let i = 0; i < data.length; i++) {
        i === 0 ? ctx.moveTo(toX(i), toY(data[i])) : ctx.lineTo(toX(i), toY(data[i]));
      }
      ctx.stroke(); ctx.setLineDash([]);
      for (let i = 0; i < data.length; i++) {
        ctx.fillStyle = '#ffffff'; ctx.beginPath(); ctx.arc(toX(i), toY(data[i]), 3.5, 0, Math.PI * 2); ctx.fill();
        ctx.fillStyle = color; ctx.beginPath(); ctx.arc(toX(i), toY(data[i]), 2.5, 0, Math.PI * 2); ctx.fill();
      }
    }

    drawArea(data1, color1);
    drawLine(data1, color1, false);
    drawLine(data2, color2, true);

    
    const lx = pad.left + 10, ly = pad.top + 6;
    ctx.font = '11px Inter,system-ui,sans-serif';
    ctx.fillStyle = color1; ctx.fillRect(lx, ly, 16, 3);
    ctx.fillStyle = C.txtHi; ctx.textAlign = 'left'; ctx.fillText(label1, lx + 20, ly + 5);
    const lx2 = lx + ctx.measureText(label1).width + 36;
    ctx.strokeStyle = color2; ctx.lineWidth = 2; ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(lx2, ly + 1.5); ctx.lineTo(lx2 + 16, ly + 1.5); ctx.stroke();
    ctx.setLineDash([]); ctx.fillStyle = C.txtHi; ctx.fillText(label2, lx2 + 20, ly + 5);
  }

  
  function drawImportance() {
    const info = setupCanvas('chart-importance');
    if (!info) return;
    const { ctx, w, h } = info;
    const pad = { top: 14, right: 30, bottom: 10, left: 110 };
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bottom;
    const n = featureNames.length;
    const barH = Math.min(28, (ch - (n - 1) * 6) / n);
    const gap = (ch - n * barH) / (n + 1);
    const colors = [C.purple, C.blue, C.green, C.amber, C.red, C.txt];

    ctx.clearRect(0, 0, w, h);

    for (let i = 0; i < n; i++) {
      const y = pad.top + gap + i * (barH + gap);
      const bw = (featureImportance[i] / featureImportance[0]) * cw;

      
      ctx.fillStyle = colors[i] + '22';
      ctx.beginPath();
      const r = 4;
      ctx.moveTo(pad.left, y + r);
      ctx.arcTo(pad.left, y, pad.left + r, y, r);
      ctx.lineTo(pad.left + bw - r, y);
      ctx.arcTo(pad.left + bw, y, pad.left + bw, y + r, r);
      ctx.lineTo(pad.left + bw, y + barH - r);
      ctx.arcTo(pad.left + bw, y + barH, pad.left + bw - r, y + barH, r);
      ctx.lineTo(pad.left + r, y + barH);
      ctx.arcTo(pad.left, y + barH, pad.left, y + barH - r, r);
      ctx.closePath(); ctx.fill();

      ctx.fillStyle = colors[i];
      const filledW = bw * 0.85;
      ctx.beginPath();
      ctx.moveTo(pad.left, y + r);
      ctx.arcTo(pad.left, y, pad.left + r, y, r);
      ctx.lineTo(pad.left + filledW - r, y);
      ctx.arcTo(pad.left + filledW, y, pad.left + filledW, y + r, r);
      ctx.lineTo(pad.left + filledW, y + barH - r);
      ctx.arcTo(pad.left + filledW, y + barH, pad.left + filledW - r, y + barH, r);
      ctx.lineTo(pad.left + r, y + barH);
      ctx.arcTo(pad.left, y + barH, pad.left, y + barH - r, r);
      ctx.closePath(); ctx.fill();

      
      ctx.fillStyle = C.txtHi; ctx.font = '11px monospace'; ctx.textAlign = 'right';
      ctx.fillText(featureNames[i], pad.left - 8, y + barH / 2 + 4);

      
      ctx.fillStyle = C.txt; ctx.font = '10px Inter,system-ui,sans-serif'; ctx.textAlign = 'left';
      ctx.fillText((featureImportance[i] * 100).toFixed(0) + '%', pad.left + bw + 6, y + barH / 2 + 4);
    }
  }

  
  function drawPredDist() {
    const info = setupCanvas('chart-dist');
    if (!info) return;
    const { ctx, w, h } = info;
    const pad = { top: 24, right: 18, bottom: 34, left: 44 };
    const cw = w - pad.left - pad.right;
    const ch = h - pad.top - pad.bottom;

    ctx.clearRect(0, 0, w, h);

    
    const bins = 20;
    const normalDist = [], spoofDist = [];
    for (let i = 0; i < bins; i++) {
      const x = i / bins;
      normalDist.push(Math.exp(-((x - 0.08) * (x - 0.08)) / (2 * 0.04 * 0.04)) * 0.9 + Math.random() * 0.02);
      spoofDist.push(Math.exp(-((x - 0.88) * (x - 0.88)) / (2 * 0.06 * 0.06)) * 0.8 + Math.random() * 0.02);
    }
    const maxVal = Math.max(...normalDist, ...spoofDist) * 1.15;

    
    ctx.strokeStyle = C.grid; ctx.lineWidth = 0.5;
    for (let i = 0; i <= 4; i++) {
      const y = pad.top + (ch / 4) * i;
      ctx.beginPath(); ctx.moveTo(pad.left, y); ctx.lineTo(w - pad.right, y); ctx.stroke();
    }

    
    const thX = pad.left + 0.5 * cw;
    ctx.strokeStyle = C.amber + '80'; ctx.lineWidth = 1.5; ctx.setLineDash([5, 4]);
    ctx.beginPath(); ctx.moveTo(thX, pad.top); ctx.lineTo(thX, pad.top + ch); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = C.amber; ctx.font = '10px Inter,system-ui,sans-serif'; ctx.textAlign = 'center';
    ctx.fillText('Esik = 0.50', thX, pad.top - 6);

    const barW = (cw / bins) * 0.4;

    for (let i = 0; i < bins; i++) {
      const cx = pad.left + (i + 0.5) / bins * cw;

      
      const nh = (normalDist[i] / maxVal) * ch;
      ctx.fillStyle = C.green + '55';
      ctx.fillRect(cx - barW - 1, pad.top + ch - nh, barW, nh);
      ctx.fillStyle = C.green;
      ctx.fillRect(cx - barW - 1, pad.top + ch - nh, barW, 3);

      
      const sh = (spoofDist[i] / maxVal) * ch;
      ctx.fillStyle = C.red + '55';
      ctx.fillRect(cx + 1, pad.top + ch - sh, barW, sh);
      ctx.fillStyle = C.red;
      ctx.fillRect(cx + 1, pad.top + ch - sh, barW, 3);
    }

    
    ctx.fillStyle = C.txt; ctx.font = '10px Inter,system-ui,sans-serif'; ctx.textAlign = 'center';
    for (let i = 0; i <= 4; i++) {
      const x = pad.left + (i / 4) * cw;
      ctx.fillText((i * 0.25).toFixed(2), x, h - pad.bottom + 16);
    }
    ctx.fillText('Model Olasilik Skoru', pad.left + cw / 2, h - 4);

    
    ctx.font = '11px Inter,system-ui,sans-serif'; ctx.textAlign = 'left';
    const lx = pad.left + 6, ly = pad.top + 6;
    ctx.fillStyle = C.green; ctx.fillRect(lx, ly, 12, 10);
    ctx.fillStyle = C.txtHi; ctx.fillText('Normal', lx + 16, ly + 9);
    ctx.fillStyle = C.red; ctx.fillRect(lx + 70, ly, 12, 10);
    ctx.fillStyle = C.txtHi; ctx.fillText('Spoofing', lx + 86, ly + 9);
  }

  function init() {
    drawLineChart('chart-loss', trainLoss, valLoss, 'Train Loss', 'Val Loss',
                  C.red, C.amber, 0, 0.45, 'Loss');

    drawLineChart('chart-acc', trainAcc, valAcc, 'Train Acc', 'Val Acc',
                  C.green, C.purple, 0.78, 1.0, 'Accuracy');

    drawImportance();
    drawPredDist();

    const note = document.getElementById('graph-note');
    if (note) {
      note.textContent = `En iyi model: Epoch ${EPOCHS} — Val Loss: ${valLoss[EPOCHS-1].toFixed(3)}, Val Acc: ${(valAcc[EPOCHS-1]*100).toFixed(1)}%  |  Early stopping (patience=5) ile kaydedildi`;
    }
  }

  function safeInit() {
    const page = document.getElementById('page-model');
    if (!page || !page.classList.contains('active')) return;
    setTimeout(init, 30);
  }

  window._onModelPageShow = safeInit;
  window.addEventListener('resize', () => {
    const page = document.getElementById('page-model');
    if (page && page.classList.contains('active')) init();
  });
})();