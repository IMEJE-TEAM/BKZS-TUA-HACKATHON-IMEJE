(function () {
  'use strict';

  const $ = id => document.getElementById(id);
  const MAX_ENTRIES = 600;
  const SAMPLE_EVERY = 40;
  const TOTAL_ROWS = 30000;

  let DATA = null;
  let cursor = 0;
  let running = false;
  let paused = false;
  let speed = 5;
  let rafId = null;
  let lastFrame = 0;
  let autoScroll = true;
  let stats = { total: 0, spoof: 0, normal: 0, transitions: 0 };
  let prevLabel = -1;
  let allEntries = [];
  let lastSimState = null;

  function rowToTime(idx) {
    const total = DATA ? DATA.totalRows : TOTAL_ROWS;
    const sec = Math.floor((idx / total) * 86400);
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    return { h, m, s, str: String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0') };
  }

  function makeEntry(i) {
    const label = DATA.labels[i];
    const isTransition = label !== prevLabel && prevLabel !== -1;
    const type = label === 1 ? 'spoof' : 'normal';
    const prob = DATA.probs[i];
    return {
      idx: i, time: rowToTime(i), label, type, prob, isTransition,
      prRes:    DATA.data.mean_prRes     ? DATA.data.mean_prRes[i]     : 0,
      stdPrRes: DATA.data.std_prRes      ? DATA.data.std_prRes[i]      : 0,
      maxPrRes: DATA.data.max_prRes      ? DATA.data.max_prRes[i]      : 0,
      cno:      DATA.data.mean_cno       ? DATA.data.mean_cno[i]       : 0,
      stdCno:   DATA.data.std_cno        ? DATA.data.std_cno[i]        : 0,
      cnoElev:  DATA.data.cno_elev_ratio ? DATA.data.cno_elev_ratio[i] : 0,
    };
  }

  function createEntryEl(entry) {
    const el = document.createElement('div');
    el.className = 'tl-entry';
    const badgeCls  = entry.isTransition ? 'transition' : entry.type;
    const badgeText = entry.isTransition ? 'TRANS' : (entry.type === 'spoof' ? 'SPOOF' : 'NORML');
    const probCls   = entry.prob >= 0.5 ? 'tl-prob-high' : 'tl-prob-low';
    el.innerHTML =
      `<span class="tl-time">${entry.time.str}</span>` +
      `<span class="tl-badge ${badgeCls}">[${badgeText}]</span> ` +
      `<span class="tl-data">` +
        `<span class="tl-key">idx=</span><span class="tl-val">${entry.idx}</span> ` +
        `<span class="tl-key">prob=</span><span class="${probCls}">${(entry.prob * 100).toFixed(1)}%</span> ` +
        `<span class="tl-key">prRes=</span><span class="tl-val">${entry.prRes.toFixed(2)}</span> ` +
        `<span class="tl-key">cno=</span><span class="tl-val">${entry.cno.toFixed(1)}</span> ` +
        `<span class="tl-key">elev=</span><span class="tl-val">${entry.cnoElev.toFixed(3)}</span>` +
      `</span>`;
    return el;
  }

  function passesFilter(entry) {
    const fType = $('filter-type').value;
    if (fType === 'spoof' && entry.type !== 'spoof') return false;
    if (fType === 'normal' && entry.type !== 'normal') return false;
    if (fType === 'transition' && !entry.isTransition) return false;
    const search = ($('filter-search').value || '').toLowerCase().trim();
    if (search) {
      const haystack = `${entry.time.str} ${entry.type} ${entry.idx} ${entry.prob.toFixed(3)} ${entry.prRes.toFixed(2)} ${entry.cno.toFixed(1)}`.toLowerCase();
      if (!haystack.includes(search)) return false;
    }
    return true;
  }

  function updateSummary() {
    $('lsc-total').textContent  = stats.total.toLocaleString();
    $('lsc-spoof').textContent  = stats.spoof.toLocaleString();
    $('lsc-normal').textContent = stats.normal.toLocaleString();
    const rate = stats.total > 0 ? ((stats.spoof / stats.total) * 100).toFixed(1) : '0.0';
    $('lsc-rate').textContent = rate + '%';
    const total = DATA ? DATA.totalRows : TOTAL_ROWS;
    const pct = DATA ? Math.min(100, (cursor / total) * 100).toFixed(0) : 0;
    $('lsc-bar-total').style.width  = pct + '%';
    $('lsc-bar-spoof').style.width  = (stats.total > 0 ? (stats.spoof  / stats.total * 100) : 0).toFixed(0) + '%';
    $('lsc-bar-normal').style.width = (stats.total > 0 ? (stats.normal / stats.total * 100) : 0).toFixed(0) + '%';
    $('lsc-bar-rate').style.width   = rate + '%';
    $('timeline-progress').textContent = pct + '%';
    $('log-count').textContent = allEntries.length.toLocaleString() + ' kayit';
  }

  function processBatch() {
    if (!DATA) return;
    const total = DATA.totalRows;
    if (cursor >= total) return;
    const end = Math.min(cursor + speed, total);
    const stream = $('log-stream');
    let addedAny = false;
    for (let i = cursor; i < end; i++) {
      const label = DATA.labels[i];
      const isTransition = label !== prevLabel && prevLabel !== -1;
      stats.total++;
      if (label === 1) stats.spoof++; else stats.normal++;
      if (isTransition) stats.transitions++;
      const showEntry = isTransition || (i - cursor) % SAMPLE_EVERY === 0 || i === end - 1;
      if (showEntry) {
        const entry = makeEntry(i);
        prevLabel = label;
        allEntries.push(entry);
        if (passesFilter(entry)) {
          const el = createEntryEl(entry);
          stream.appendChild(el);
          addedAny = true;
          while (stream.querySelectorAll('.tl-entry').length > MAX_ENTRIES) {
            const first = stream.querySelector('.tl-entry');
            if (first) first.remove(); else break;
          }
        }
      } else {
        prevLabel = label;
      }
    }
    cursor = end;
    if (addedAny && autoScroll) stream.scrollTop = stream.scrollHeight;
    const emp = $('stream-empty');
    if (emp && allEntries.length > 0) emp.style.display = 'none';
    updateSummary();
  }

  function tick(ts) {
    if (!running || paused) { rafId = null; return; }
    const interval = 1000 / 30;
    if (ts - lastFrame >= interval) { lastFrame = ts; processBatch(); }
    if (running) rafId = requestAnimationFrame(tick);
  }

  function startStream() {
    if (running) return;
    DATA = (window._bkzsSim && window._bkzsSim.data) ? window._bkzsSim.data : null;
    if (!DATA) return;
    running = true;
    paused = false;
    $('stream-dot').classList.add('active');
    const emp = $('stream-empty');
    if (emp) emp.style.display = 'none';
    lastFrame = performance.now();
    rafId = requestAnimationFrame(tick);
  }

  function pauseStream(state) {
    paused = state;
    if (!paused && running) { lastFrame = performance.now(); rafId = requestAnimationFrame(tick); }
  }

  function resetStream() {
    running = false; paused = false; cursor = 0; prevLabel = -1;
    stats = { total: 0, spoof: 0, normal: 0, transitions: 0 };
    allEntries = [];
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;
    DATA = null;
    const stream = $('log-stream');
    stream.innerHTML = '';
    const emp = document.createElement('div');
    emp.className = 'term-empty'; emp.id = 'stream-empty';
    emp.innerHTML = '<span class="term-prompt">$</span><p>Simulasyon basladiginda loglar otomatik olarak akacak...</p>';
    stream.appendChild(emp);
    $('stream-dot').classList.remove('active');
    updateSummary();
  }

  function pollSimState() {
    const sim = window._bkzsSim;
    if (sim) {
      if (sim.running && !running) {
        speed = sim.speed || 5;
        startStream();
      }
      if (running) {
        if (sim.speed && sim.speed !== speed) speed = sim.speed;
        if (sim.paused !== paused) pauseStream(sim.paused);
      }
      if (!sim.running && running) {
        running = false;
        $('stream-dot').classList.remove('active');
      }
      return;
    }
    try {
      const raw = localStorage.getItem('bkzs_sim');
      if (!raw) return;
      const state = JSON.parse(raw);
      if (state.running && !running && state.timestamp > (lastSimState ? lastSimState.timestamp : 0)) {
        speed = state.speed || 5;
        startStream();
      }
      if (running) {
        if (state.speed && state.speed !== speed) speed = state.speed;
        if (state.paused !== paused) pauseStream(state.paused);
      }
      lastSimState = state;
    } catch(e) {}
  }
  setInterval(pollSimState, 300);

  function exportCSV() {
    if (allEntries.length === 0) return;
    const header = 'idx,time,type,prob,mean_prRes,std_prRes,max_prRes,mean_cno,std_cno,cno_elev_ratio,is_transition\n';
    let csv = header;
    for (const e of allEntries) {
      csv += `${e.idx},${e.time.str},${e.type},${e.prob.toFixed(4)},${e.prRes.toFixed(4)},${e.stdPrRes.toFixed(4)},${e.maxPrRes.toFixed(4)},${e.cno.toFixed(4)},${e.stdCno.toFixed(4)},${e.cnoElev.toFixed(4)},${e.isTransition ? 1 : 0}\n`;
    }
    const blob = new Blob([csv], { type: 'text/csv;charset=utf-8;' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'bkzs_logs_' + new Date().toISOString().slice(0, 10) + '.csv';
    a.click();
    URL.revokeObjectURL(url);
  }

  function clearEntries() {
    const stream = $('log-stream');
    stream.querySelectorAll('.tl-entry').forEach(el => el.remove());
  }

  function onFilterChange() {
    const stream = $('log-stream');
    stream.querySelectorAll('.tl-entry').forEach(el => el.remove());
    const emp = $('stream-empty');
    if (emp) emp.style.display = 'none';
    const startIdx = Math.max(0, allEntries.length - MAX_ENTRIES);
    for (let i = startIdx; i < allEntries.length; i++) {
      if (passesFilter(allEntries[i])) stream.appendChild(createEntryEl(allEntries[i]));
    }
    if (autoScroll) stream.scrollTop = stream.scrollHeight;
  }

  $('btn-export').addEventListener('click', exportCSV);
  $('btn-clear').addEventListener('click', clearEntries);
  $('filter-type').addEventListener('change', onFilterChange);
  $('filter-search').addEventListener('input', onFilterChange);
  $('chk-autoscroll').addEventListener('change', function () { autoScroll = this.checked; });
})();
