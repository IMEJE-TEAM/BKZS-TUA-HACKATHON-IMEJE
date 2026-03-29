/* ======================================================================
   IMEJE-BKZS — Logs Page Script (logs.js)
   VS Code terminal-style streaming log engine
   ====================================================================== */

(function () {
  'use strict';

  const $ = id => document.getElementById(id);
  const BATCH_SIZE = 8;
  const MAX_ENTRIES = 600;
  const SAMPLE_EVERY = 40;

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

  /* ── Load data ──────────────────────────────────────────────────── */
  async function loadData() {
    $('loading-logs').style.display = 'flex';
    try {
      const res = await fetch('data/simulation_data.json');
      DATA = await res.json();
      $('loading-logs').style.display = 'none';
    } catch (e) {
      $('loading-logs').innerHTML = '<p style="color:var(--red)">Veri yuklenemedi!</p>';
    }
  }

  /* ── Time mapping ───────────────────────────────────────────────── */
  function rowToTime(idx) {
    const totalSec = 24 * 3600;
    const sec = Math.floor((idx / DATA.totalRows) * totalSec);
    const h = Math.floor(sec / 3600);
    const m = Math.floor((sec % 3600) / 60);
    const s = sec % 60;
    return { h, m, s, str: String(h).padStart(2, '0') + ':' + String(m).padStart(2, '0') + ':' + String(s).padStart(2, '0') };
  }

  /* ── Create a log entry object ──────────────────────────────────── */
  function makeEntry(i) {
    const label = DATA.labels[i];
    const isTransition = label !== prevLabel && prevLabel !== -1;
    const type = label === 1 ? 'spoof' : 'normal';
    const prob = DATA.probs[i];
    return {
      idx: i,
      time: rowToTime(i),
      label, type, prob,
      isTransition,
      prRes: DATA.data.mean_prRes ? DATA.data.mean_prRes[i] : 0,
      stdPrRes: DATA.data.std_prRes ? DATA.data.std_prRes[i] : 0,
      maxPrRes: DATA.data.max_prRes ? DATA.data.max_prRes[i] : 0,
      cno: DATA.data.mean_cno ? DATA.data.mean_cno[i] : 0,
      stdCno: DATA.data.std_cno ? DATA.data.std_cno[i] : 0,
      cnoElev: DATA.data.cno_elev_ratio ? DATA.data.cno_elev_ratio[i] : 0,
    };
  }

  /* ── DOM: terminal-style entry ──────────────────────────────────── */
  function createEntryEl(entry) {
    const el = document.createElement('div');
    el.className = 'tl-entry';

    const badgeCls = entry.isTransition ? 'transition' : entry.type;
    const badgeText = entry.isTransition ? 'TRANS' : (entry.type === 'spoof' ? 'SPOOF' : 'NORML');
    const probCls = entry.prob >= 0.5 ? 'tl-prob-high' : 'tl-prob-low';

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

  /* ── Filters ────────────────────────────────────────────────────── */
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

  /* ── Update summary cards ───────────────────────────────────────── */
  function updateSummary() {
    $('lsc-total').textContent = stats.total.toLocaleString();
    $('lsc-spoof').textContent = stats.spoof.toLocaleString();
    $('lsc-normal').textContent = stats.normal.toLocaleString();
    const rate = stats.total > 0 ? ((stats.spoof / stats.total) * 100).toFixed(1) : '0.0';
    $('lsc-rate').textContent = rate + '%';

    const pct = DATA ? ((cursor / DATA.totalRows) * 100).toFixed(0) : 0;
    $('lsc-bar-total').style.width = pct + '%';
    $('lsc-bar-spoof').style.width = (stats.total > 0 ? (stats.spoof / stats.total * 100) : 0).toFixed(0) + '%';
    $('lsc-bar-normal').style.width = (stats.total > 0 ? (stats.normal / stats.total * 100) : 0).toFixed(0) + '%';
    $('lsc-bar-rate').style.width = rate + '%';

    $('timeline-progress').textContent = pct + '%';
    $('log-count').textContent = allEntries.length.toLocaleString() + ' kayit';
  }

  /* ── Stream Engine ──────────────────────────────────────────────── */
  function processBatch() {
    if (!DATA) return;
    const total = DATA.totalRows;
    const end = Math.min(cursor + BATCH_SIZE * speed, total);
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

    if (addedAny && autoScroll) {
      stream.scrollTop = stream.scrollHeight;
    }

    const emp = $('stream-empty');
    if (emp && allEntries.length > 0) emp.style.display = 'none';

    updateSummary();

    if (cursor >= total) {
      running = false;
      updateButtons();
      $('stream-dot').classList.remove('active');
    }
  }

  function tick(ts) {
    if (!running || paused) { rafId = null; return; }
    const interval = 1000 / 30;
    if (ts - lastFrame >= interval) {
      lastFrame = ts;
      processBatch();
    }
    if (running) rafId = requestAnimationFrame(tick);
  }

  function startStream() {
    if (!DATA) return;
    running = true;
    paused = false;
    $('stream-dot').classList.add('active');
    updateButtons();
    lastFrame = performance.now();
    rafId = requestAnimationFrame(tick);
  }

  function pauseStream() {
    paused = !paused;
    updateButtons();
    if (!paused && running) {
      lastFrame = performance.now();
      rafId = requestAnimationFrame(tick);
    }
  }

  function resetStream() {
    running = false;
    paused = false;
    cursor = 0;
    prevLabel = -1;
    stats = { total: 0, spoof: 0, normal: 0, transitions: 0 };
    allEntries = [];
    if (rafId) cancelAnimationFrame(rafId);
    rafId = null;

    const stream = $('log-stream');
    stream.innerHTML = '';
    const emp = document.createElement('div');
    emp.className = 'term-empty';
    emp.id = 'stream-empty';
    emp.innerHTML = '<span class="term-prompt">$</span><p>Log sifirlandi. Tekrar baslatmak icin Baslat butonuna basin.</p>';
    stream.appendChild(emp);

    $('stream-dot').classList.remove('active');
    updateSummary();
    updateButtons();
  }

  function updateButtons() {
    $('btn-play').disabled = running && !paused;
    $('btn-pause').disabled = !running;
    $('btn-pause').textContent = paused ? 'Devam Et' : 'Duraklat';
  }

  /* ── CSV Export ─────────────────────────────────────────────────── */
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
    const entries = stream.querySelectorAll('.tl-entry');
    entries.forEach(el => el.remove());
  }

  function onFilterChange() {
    const stream = $('log-stream');
    const entries = stream.querySelectorAll('.tl-entry');
    entries.forEach(el => el.remove());
    const emp = $('stream-empty');
    if (emp) emp.style.display = 'none';

    const startIdx = Math.max(0, allEntries.length - MAX_ENTRIES);
    for (let i = startIdx; i < allEntries.length; i++) {
      if (passesFilter(allEntries[i])) {
        stream.appendChild(createEntryEl(allEntries[i]));
      }
    }
    if (autoScroll) stream.scrollTop = stream.scrollHeight;
  }

  /* ── Event Listeners ────────────────────────────────────────────── */
  $('btn-play').addEventListener('click', startStream);
  $('btn-pause').addEventListener('click', pauseStream);
  $('btn-export').addEventListener('click', exportCSV);
  $('btn-clear').addEventListener('click', clearEntries);
  $('filter-type').addEventListener('change', onFilterChange);
  $('filter-search').addEventListener('input', onFilterChange);
  $('chk-autoscroll').addEventListener('change', function () { autoScroll = this.checked; });
  $('log-speed').addEventListener('change', function () { speed = parseInt(this.value) || 5; });

  document.addEventListener('keydown', function (e) {
    if (e.target.tagName === 'INPUT' || e.target.tagName === 'SELECT') return;
    if (e.code === 'Space') { e.preventDefault(); if (running) pauseStream(); else startStream(); }
    if (e.code === 'KeyR') { e.preventDefault(); resetStream(); }
  });

  /* ── Init ───────────────────────────────────────────────────────── */
  loadData().then(() => {
    updateButtons();
  });

})();
