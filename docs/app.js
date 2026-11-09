// NBA Player Projections: renders docs/data/projections.json (built by build_site.py).

const TEAM_COLORS = {
  ATL: '#E03A3E', BOS: '#007A33', BKN: '#2b2b2b', CHA: '#1D1160', CHI: '#CE1141', CLE: '#860038',
  DAL: '#00538C', DEN: '#0E2240', DET: '#C8102E', GSW: '#1D428A', HOU: '#CE1141', IND: '#002D62',
  LAC: '#C8102E', LAL: '#552583', MEM: '#5D76A9', MIA: '#98002E', MIL: '#00471B', MIN: '#0C2340',
  NOP: '#0C2340', NYK: '#006BB6', OKC: '#007AC1', ORL: '#0077C0', PHI: '#006BB6', PHX: '#1D1160',
  POR: '#E03A3E', SAC: '#5A2D81', SAS: '#6b6e70', TOR: '#CE1141', UTA: '#002B5C', WAS: '#002B5C',
};
const STATS = ['PTS', 'REB', 'AST'];
const STAT_NAMES = { PTS: 'Points', REB: 'Rebounds', AST: 'Assists' };
const BENCH_MINUTES = 15;   // players averaging fewer minutes are hidden behind "Show bench"

const state = { data: null, day: null, gameId: null, sort: 'PTS', bench: {}, active: 0, chartStat: 'PTS', lastFocus: null };
const $ = (id) => document.getElementById(id);

const esc = (s) => String(s ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
const fmt = (v) => (v == null ? '–' : v.toFixed(1));
const simplify = (s) => s.normalize('NFKD').replace(/[̀-ͯ]/g, '').toLowerCase();
const lastName = (name) => name.split(' ').slice(1).join(' ') || name;
const badge = (abbr) => `<span class="badge" style="background:${TEAM_COLORS[abbr] || '#555'}">${esc(abbr)}</span>`;

function localDate(iso) { return new Date(iso + 'T12:00:00'); }
function dayLabel(iso) {
  const d = localDate(iso);
  const today = new Date(); today.setHours(12, 0, 0, 0);
  const diff = Math.round((d - today) / 86400000);
  const short = d.toLocaleDateString(undefined, { weekday: 'short', month: 'short', day: 'numeric' });
  if (diff === 0) return `Today · ${short}`;
  if (diff === 1) return `Tomorrow · ${short}`;
  return short;
}
function tipoff(game) {
  return new Date(game.tipoff_utc).toLocaleTimeString(undefined, { hour: 'numeric', minute: '2-digit' });
}
function timeAgo(iso) {
  const minutes = Math.round((Date.now() - new Date(iso)) / 60000);
  if (minutes < 60) return `${Math.max(minutes, 1)} min ago`;
  const hours = Math.round(minutes / 60);
  if (hours < 48) return `${hours} hour${hours === 1 ? '' : 's'} ago`;
  return `${Math.round(hours / 24)} days ago`;
}
function delta(proj, base) {
  if (proj == null || base == null) return '<span class="delta flat">new</span>';
  const d = proj - base;
  if (Math.abs(d) < 0.5) return '<span class="delta flat">– same</span>';
  return d > 0 ? `<span class="delta up">▲ ${d.toFixed(1)}</span>` : `<span class="delta down">▼ ${(-d).toFixed(1)}</span>`;
}
function game(id) { return state.data.games.find((g) => g.id === id); }
function days() { return [...new Set(state.data.games.map((g) => g.date))]; }

// ---------- Days and game cards ----------
function renderDays() {
  $('days').innerHTML = days().map((d) => {
    const n = state.data.games.filter((g) => g.date === d).length;
    return `<button role="tab" data-day="${d}" aria-selected="${d === state.day}">${esc(dayLabel(d))}<span class="count">${n}</span></button>`;
  }).join('');
}

function renderGames() {
  const games = state.data.games.filter((g) => g.date === state.day);
  if (!games.length) {
    $('game-grid').innerHTML = '<div class="empty-state">No upcoming games in the next few days. Check back soon.</div>';
    return;
  }
  $('game-grid').innerHTML = games.map((g) => {
    const outCount = g.teams[g.away.abbr].out.length + g.teams[g.home.abbr].out.length;
    const row = (side) => {
      const team = g.teams[side.abbr];
      const top = team.players[0];
      return `<div class="matchup-row">
        <span class="team-label">${badge(side.abbr)}<span class="name">${esc(side.short || side.name)}</span></span>
        ${top ? `<span class="top-scorer">${esc(lastName(top.name))} ${fmt(top.proj.PTS)}</span>` : ''}
      </div>`;
    };
    return `<button class="game-card" data-game="${g.id}" aria-pressed="${g.id === state.gameId}">
      <div class="time"><span>${tipoff(g)}</span>${outCount ? `<span class="pill out">${outCount} out</span>` : ''}</div>
      ${row(g.away)}${row(g.home)}
    </button>`;
  }).join('');
}

// ---------- One game ----------
function renderGame() {
  const g = game(state.gameId);
  if (!g) { $('game').hidden = true; return; }
  $('game').hidden = false;
  $('game-when').textContent = `${dayLabel(g.date)} · ${tipoff(g)}`;
  $('game-title').textContent = `${g.away.name} @ ${g.home.name}`;
  const miss = state.data.typical_miss;
  $('typical-miss').textContent = `Typical miss: ±${miss.PTS} pts, ±${miss.REB} reb, ±${miss.AST} ast`;

  const sides = [g.away, g.home];
  $('team-switch').innerHTML = sides.map((s, i) =>
    `<button role="tab" data-team="${i}" aria-selected="${i === state.active}">${esc(s.abbr)}</button>`).join('');
  $('teams').dataset.active = state.active;
  $('teams').innerHTML = sides.map((s, i) => teamPanel(g, s, i === 1)).join('');
}

function teamPanel(g, side, home) {
  const team = g.teams[side.abbr];
  const key = `${g.id}-${side.abbr}`;
  const showBench = state.bench[key];
  const sorted = [...team.players].sort((a, b) => (b.proj[state.sort] ?? 0) - (a.proj[state.sort] ?? 0));
  const main = sorted.filter((p) => (p.min_last10 ?? 0) >= BENCH_MINUTES);
  const bench = sorted.filter((p) => (p.min_last10 ?? 0) < BENCH_MINUTES);
  const shown = showBench ? sorted : main;

  const head = STATS.map((s) => `<th scope="col"><button data-sort="${s}" ${s === state.sort ? 'aria-sort="descending"' : ''}>${s}${s === state.sort ? ' ↓' : ''}</button></th>`).join('');
  const rows = shown.map((p) => {
    const chips = [];
    if (p.status === 'Day-To-Day') chips.push('<span class="pill gtd">Questionable</span>');
    if (p.rest_days === 1) chips.push('<span class="pill b2b">Back-to-back</span>');
    return `<tr data-player="${p.id}" data-team="${side.abbr}" tabindex="0">
      <td><span class="pname">${esc(p.name)}</span><span class="psub">${p.min_last10 != null ? `${Math.round(p.min_last10)} min` : ''} ${chips.join('')}</span></td>
      ${STATS.map((s) => `<td><span class="stat">${fmt(p.proj[s])}</span>${delta(p.proj[s], p.last10[s])}</td>`).join('')}
    </tr>`;
  }).join('');

  const benchButton = bench.length
    ? `<button class="bench-toggle" data-bench="${key}">${showBench ? 'Hide bench' : `Show bench (${bench.length} more)`}</button>` : '';
  const out = team.out.length
    ? `<p class="out-box"><b>Out:</b> ${team.out.map((o) => `${esc(o.name)} (${Math.round(o.min)} min)`).join(', ')}</p>`
    : '<p class="out-box none">No rotation players listed out.</p>';

  return `<section class="team-panel" style="--team:${TEAM_COLORS[side.abbr] || 'var(--accent)'}">
    <header>${badge(side.abbr)}<h3>${esc(side.name)}</h3><span class="where">${home ? 'Home' : 'Away'}</span></header>
    <table class="ptable"><thead><tr><th scope="col">Player</th>${head}</tr></thead><tbody>${rows}</tbody></table>
    ${benchButton}${out}
  </section>`;
}

function selectGame(id, { scroll = true } = {}) {
  const g = game(id);
  if (!g) return;
  state.gameId = id;
  state.day = g.date;
  state.active = 0;
  history.replaceState(null, '', `#game-${id}`);
  renderDays(); renderGames(); renderGame();
  if (scroll) $('game').scrollIntoView({ behavior: 'smooth', block: 'start' });
}

// ---------- Player details ----------
function openPlayer(gameId, abbr, playerId) {
  const g = game(gameId);
  const p = g.teams[abbr].players.find((x) => x.id === playerId);
  if (!p) return;
  const opp = abbr === g.home.abbr ? g.away : g.home;
  const home = abbr === g.home.abbr;
  const miss = state.data.typical_miss;
  state.lastFocus = document.activeElement;
  state.player = { g, p, abbr, opp };

  const statBlocks = STATS.map((s) => {
    const lo = Math.max(0, p.proj[s] - miss[s]), hi = p.proj[s] + miss[s];
    return `<div class="pm-stat"><div class="label">${STAT_NAMES[s]}</div><div class="big">${fmt(p.proj[s])}</div>
      <div class="range">Usually ${lo.toFixed(0)}–${hi.toFixed(0)}</div>${delta(p.proj[s], p.last10[s])}</div>`;
  }).join('');

  const compare = (label, values, cls = '') => `<tr class="${cls}"><td>${label}</td>${STATS.map((s) => `<td>${fmt(values[s])}</td>`).join('')}</tr>`;
  const vsLabel = p.vs_opp_games ? `vs. ${esc(opp.abbr)} (${p.vs_opp_games} game${p.vs_opp_games === 1 ? '' : 's'})` : `vs. ${esc(opp.abbr)} (no games yet)`;

  const teammatesOut = g.teams[abbr].out;
  const oppOut = g.teams[opp.abbr].out;
  const context = [
    `Averaging ${p.min_last10 != null ? Math.round(p.min_last10) : '–'} minutes over the last 10 games.`,
    p.rest_days == null ? null : p.rest_days === 1 ? 'Playing on the second night of a back-to-back.' : `${p.rest_days} days of rest before this game.`,
    teammatesOut.length ? `Teammates out: ${teammatesOut.map((o) => esc(o.name)).join(', ')}. Their minutes and shots go to the rest of the roster.` : 'No teammates listed out.',
    oppOut.length ? `${esc(opp.abbr)} is without ${oppOut.map((o) => esc(o.name)).join(', ')}.` : null,
  ].filter(Boolean);

  $('player-body').innerHTML = `
    <div class="pm-head">${badge(abbr)}<div><h2 id="pm-name">${esc(p.name)}</h2>
      <p class="pm-sub">${esc(abbr)} ${home ? 'vs.' : '@'} ${esc(opp.abbr)} · ${esc(dayLabel(g.date))} · ${tipoff(g)}</p></div></div>
    <div class="pm-stats">${statBlocks}</div>
    <table class="pm-compare">
      <thead><tr><th></th>${STATS.map((s) => `<th>${s}</th>`).join('')}</tr></thead>
      <tbody>${compare('Projection', p.proj, 'proj')}${compare('Last 10 games', p.last10)}${compare('This season', p.season)}${compare(vsLabel, p.vs_opp)}</tbody>
    </table>
    <div class="pm-section">
      <h3>Last ${p.recent.length} games <span class="chart-tabs">${STATS.map((s) => `<button data-chart="${s}" aria-pressed="${s === state.chartStat}">${s}</button>`).join('')}</span></h3>
      <div class="chart" id="chart"></div>
    </div>
    <div class="pm-section"><h3>Behind the number</h3><ul class="context">${context.map((c) => `<li>${c}</li>`).join('')}</ul>
      ${p.status ? `<p class="note"><b>Injury report: ${esc(p.status)}.</b> ${esc(p.note || '')} The projection assumes this player plays.</p>` : ''}
    </div>`;
  renderChart();
  $('player-modal').hidden = false;
  document.body.style.overflow = 'hidden';
  $('player-modal').querySelector('.modal-close').focus();
}

function renderChart() {
  const { p } = state.player;
  const s = state.chartStat;
  const games = p.recent;
  const W = 560, H = 190, top = 22, bottom = 34, left = 8, right = 8;
  const max = Math.max(1, p.proj[s], ...games.map((g) => g[s])) * 1.15;
  const y = (v) => top + (H - top - bottom) * (1 - v / max);
  const slot = (W - left - right) / Math.max(games.length, 1);
  const bars = games.map((g, i) => {
    const x = left + i * slot + slot * 0.18, w = slot * 0.64;
    const date = localDate(g.date).toLocaleDateString(undefined, { month: 'numeric', day: 'numeric' });
    return `<rect class="bar ${g[s] >= p.proj[s] ? 'over' : ''}" x="${x}" y="${y(g[s])}" width="${w}" height="${Math.max(0, H - bottom - y(g[s]))}" rx="3"/>
      <text class="val" x="${x + w / 2}" y="${y(g[s]) - 5}" text-anchor="middle">${g[s]}</text>
      <text x="${x + w / 2}" y="${H - bottom + 14}" text-anchor="middle">${g.home ? '' : '@'}${esc(g.opp)}</text>
      <text x="${x + w / 2}" y="${H - bottom + 27}" text-anchor="middle">${date}</text>`;
  }).join('');
  const py = y(p.proj[s]);
  $('chart').innerHTML = `<svg viewBox="0 0 ${W} ${H}" role="img" aria-label="${STAT_NAMES[s]} in the last ${games.length} games compared with the projection of ${fmt(p.proj[s])}">
    ${bars}
    <line class="proj-line" x1="${left}" x2="${W - right}" y1="${py}" y2="${py}"/>
    <text class="proj-label" x="${W - right}" y="${py - 6}" text-anchor="end">Projection ${fmt(p.proj[s])}</text>
  </svg>`;
  document.querySelectorAll('[data-chart]').forEach((b) => b.setAttribute('aria-pressed', b.dataset.chart === s));
}

function closePlayer() {
  $('player-modal').hidden = true;
  document.body.style.overflow = '';
  if (state.lastFocus) state.lastFocus.focus();
}

// ---------- Search ----------
let searchIndex = [];
let searchHits = [];
let searchActive = -1;

function buildSearch() {
  const seen = new Set();
  for (const g of state.data.games) {
    for (const side of [g.away, g.home]) {
      const opp = side === g.home ? g.away : g.home;
      const team = g.teams[side.abbr];
      // Injured players are searchable too, so people can see why there's no projection
      for (const [list, out] of [[team.players, false], [team.out, true]]) {
        for (const p of list) {
          if (seen.has(p.name)) continue;   // games are in date order, so this keeps each player's next game
          seen.add(p.name);
          searchIndex.push({ key: simplify(p.name), name: p.name, abbr: side.abbr, opp: opp.abbr, home: side === g.home,
                             gameId: g.id, date: g.date, id: p.id, out });
        }
      }
    }
  }
}

function renderSearch() {
  const q = simplify($('search').value.trim());
  const list = $('search-results');
  if (!q) { list.hidden = true; $('search').setAttribute('aria-expanded', 'false'); return; }
  searchHits = searchIndex.filter((p) => p.key.includes(q))
    .sort((a, b) => a.key.indexOf(q) - b.key.indexOf(q) || a.name.localeCompare(b.name)).slice(0, 8);
  searchActive = searchHits.length ? 0 : -1;
  list.innerHTML = searchHits.length
    ? searchHits.map((p, i) => `<li role="option" data-i="${i}" aria-selected="${i === searchActive}">${badge(p.abbr)}<span>${esc(p.name)}</span>
        <span class="meta">${p.out ? '<span class="pill out">Out</span> ' : ''}${p.home ? 'vs.' : '@'} ${esc(p.opp)} · ${esc(dayLabel(p.date).split(' · ')[0])}</span></li>`).join('')
    : `<li class="empty">No match. Only rotation players with a game in the next ${days().length} days are listed.</li>`;
  list.hidden = false;
  $('search').setAttribute('aria-expanded', 'true');
}

function chooseSearch(i) {
  const hit = searchHits[i];
  if (!hit) return;
  $('search').value = '';
  $('search-results').hidden = true;
  if (hit.out) {   // no projection for injured players: show their game and the "Out" list
    const g = game(hit.gameId);
    selectGame(hit.gameId);
    state.active = hit.abbr === g.home.abbr ? 1 : 0;
    renderGame();
    return;
  }
  selectGame(hit.gameId, { scroll: false });
  openPlayer(hit.gameId, hit.abbr, hit.id);
}

// ---------- Events ----------
function bindEvents() {
  $('days').addEventListener('click', (e) => {
    const b = e.target.closest('[data-day]');
    if (!b) return;
    state.day = b.dataset.day;
    renderDays(); renderGames();
  });
  $('game-grid').addEventListener('click', (e) => {
    const b = e.target.closest('[data-game]');
    if (b) selectGame(b.dataset.game);
  });
  $('team-switch').addEventListener('click', (e) => {
    const b = e.target.closest('[data-team]');
    if (!b) return;
    state.active = Number(b.dataset.team);
    renderGame();
  });
  $('teams').addEventListener('click', (e) => {
    const sort = e.target.closest('[data-sort]');
    if (sort) { state.sort = sort.dataset.sort; renderGame(); return; }
    const bench = e.target.closest('[data-bench]');
    if (bench) { state.bench[bench.dataset.bench] = !state.bench[bench.dataset.bench]; renderGame(); return; }
    const row = e.target.closest('[data-player]');
    if (row) openPlayer(state.gameId, row.dataset.team, Number(row.dataset.player));
  });
  $('teams').addEventListener('keydown', (e) => {
    const row = e.target.closest('[data-player]');
    if (row && (e.key === 'Enter' || e.key === ' ')) { e.preventDefault(); openPlayer(state.gameId, row.dataset.team, Number(row.dataset.player)); }
  });
  $('player-modal').addEventListener('click', (e) => {
    if (e.target.closest('[data-close]')) closePlayer();
    const tab = e.target.closest('[data-chart]');
    if (tab) { state.chartStat = tab.dataset.chart; renderChart(); }
  });
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && !$('player-modal').hidden) closePlayer();
  });

  const search = $('search');
  search.addEventListener('input', renderSearch);
  search.addEventListener('keydown', (e) => {
    if ($('search-results').hidden) return;
    if (e.key === 'ArrowDown' || e.key === 'ArrowUp') {
      e.preventDefault();
      if (!searchHits.length) return;
      searchActive = (searchActive + (e.key === 'ArrowDown' ? 1 : -1) + searchHits.length) % searchHits.length;
      document.querySelectorAll('#search-results [data-i]').forEach((li) => li.setAttribute('aria-selected', Number(li.dataset.i) === searchActive));
    } else if (e.key === 'Enter') {
      e.preventDefault(); chooseSearch(searchActive);
    } else if (e.key === 'Escape') {
      $('search-results').hidden = true;
    }
  });
  $('search-results').addEventListener('mousedown', (e) => {
    const li = e.target.closest('[data-i]');
    if (li) { e.preventDefault(); chooseSearch(Number(li.dataset.i)); }
  });
  search.addEventListener('blur', () => setTimeout(() => { $('search-results').hidden = true; }, 100));
}

// ---------- Start ----------
async function init() {
  try {
    const response = await fetch('data/projections.json', { cache: 'no-cache' });
    state.data = await response.json();
  } catch (err) {
    $('updated').textContent = 'Could not load projections. Try refreshing the page.';
    return;
  }
  const d = state.data;
  const through = localDate(d.data_through).toLocaleDateString(undefined, { month: 'short', day: 'numeric' });
  $('updated').textContent = `Updated ${timeAgo(d.generated_at)} · includes games through ${through} · ${d.season} season`;

  bindEvents();
  buildSearch();
  const fromLink = location.hash.startsWith('#game-') && game(location.hash.slice(6));
  state.day = fromLink ? fromLink.date : days()[0] ?? null;
  renderDays();
  renderGames();
  if (fromLink) selectGame(fromLink.id, { scroll: false });
}

init();
