'use client';
import { useEffect, useRef, useState } from 'react';
import type { Map as LeafletMap, GeoJSON as LeafletGeoJSON } from 'leaflet';
import type { FeatureCollection } from 'geojson';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import 'leaflet/dist/leaflet.css';
type ModelState = { state: string; available: boolean; epoch?: number; epochs_requested?: number };
type Facet = { id: number; pitch_deg: number; azimuth_deg: number; panel_count: number; usable_area_m2: number };
type Result = { building_id: number; geojson: FeatureCollection; panel_count: number; additional_kwp: number; annual_kwh: number | null; usable_area_m2: number; existing_pv_area_m2: number; facets: Facet[]; warnings: string[]; image: string; provisional?: boolean };
async function request<T>(path: string): Promise<T> {
  const response = await fetch(path); const body: unknown = await response.json();
  if (!response.ok) {
    const detail = body && typeof body === 'object' && 'detail' in body ? body.detail : null;
    throw new Error(typeof detail === 'string' ? detail : 'Could not complete this request.');
  }
  return body as T;
}
export default function Home() {
  const mapElement = useRef<HTMLDivElement>(null);
  const map = useRef<LeafletMap | null>(null);
  const overlay = useRef<LeafletGeoJSON | null>(null);
  const analyzeRef = useRef<(lat: number, lon: number) => void>(() => {});
  const sequence = useRef(0);
  const [models, setModels] = useState<Record<string, ModelState>>({});
  const [result, setResult] = useState<Result | null>(null);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('Click inside a house roof to analyze it.');
  const [error, setError] = useState('');
  const [address, setAddress] = useState('');
  const [searching, setSearching] = useState(false);
  const [searchResults, setSearchResults] = useState<{ label: string; lat: number; lon: number }[]>([]);
  const [confidence, setConfidence] = useState(.25);
  const [obstacleConfidence, setObstacleConfidence] = useState(.25);
  const [setback, setSetback] = useState(.6);
  const [rowGap, setRowGap] = useState(.35);
  const [selected, setSelected] = useState<[number, number] | null>(null);
  async function analyze(lat: number, lon: number) {
    const id = ++sequence.current;
    setSelected([lat, lon]); setBusy(true); setError(''); setResult(null); overlay.current?.clearLayers();
    setMessage('Finding roof faces, detecting objects and fitting panels…');
    try {
      const data = await request<Result>(`/api/analyze?lat=${lat}&lon=${lon}&confidence=${confidence}&obstacle_confidence=${obstacleConfidence}&setback=${setback}&row_gap=${rowGap}`);
      if (id !== sequence.current) return;
      setResult(data);
      const L = await import('leaflet');
      if (overlay.current) overlay.current.remove();
      overlay.current = L.geoJSON(data.geojson, {
        style: feature => {
          const kind = feature?.properties?.kind;
          return kind === 'panel' ? { color: '#66f5ba', weight: 1, fillColor: '#11bc82', fillOpacity: .55 }
            : kind === 'pv' ? { color: '#55aaff', fillColor: '#2585ef', weight: 2, fillOpacity: .6 }
            : kind === 'obstacle' ? { color: '#ff814f', fillColor: '#ff814f', weight: 2, fillOpacity: .6 }
            : kind === 'free' ? { color: '#b3e8db', weight: 0, fillOpacity: .12 }
            : { color: '#f5e06a', weight: 2, fillOpacity: .03 };
        },
        onEachFeature: (feature, layer) => {
          const p = feature.properties || {};
          const text = p.kind === 'roof' ? `Roof: ${p.pitch}° pitch, ${Math.round(p.azimuth)}° direction`
            : p.kind === 'panel' ? 'New module · 450 W · 1.762 × 1.134 m'
            : p.kind === 'free' ? 'Roof without detected PV or obstacles'
            : `${String(p.label).replaceAll('_', ' ')} · ${Math.round(p.confidence * 100)}% confidence`;
          layer.bindTooltip(text);
        },
      }).addTo(map.current!);
      map.current!.fitBounds(overlay.current.getBounds(), { padding: [45, 45], maxZoom: 21 });
      setMessage(`Building ${data.building_id} · ${data.facets.length} roof faces`);
    } catch (e) {
      if (id === sequence.current) { setError((e as Error).message); setMessage('Select another roof or try again.'); }
    } finally { if (id === sequence.current) setBusy(false); }
  }
  useEffect(() => { analyzeRef.current = (lat, lon) => { if (!busy) void analyze(lat, lon); }; });
  useEffect(() => {
    let disposed = false;
    void import('leaflet').then(L => {
      if (disposed || !mapElement.current) return;
      const instance = L.map(mapElement.current, { maxZoom: 22 }).setView([47.3917, 8.0452], 19);
      L.tileLayer('https://wmts.geo.admin.ch/1.0.0/ch.swisstopo.swissimage/default/current/3857/{z}/{x}/{y}.jpeg', {
        attribution: '© swisstopo · Sonnendach / SFOE', maxNativeZoom: 20, maxZoom: 22,
      }).addTo(instance);
      L.control.scale({ imperial: false }).addTo(instance);
      instance.on('click', event => analyzeRef.current(event.latlng.lat, event.latlng.lng));
      map.current = instance;
    }).catch(() => setError('The map could not load. Refresh the page to try again.'));
    return () => { disposed = true; map.current?.remove(); map.current = null; };
  }, []);
  useEffect(() => {
    const refresh = () => request<{ models: Record<string, ModelState> }>('/api/status').then(data => setModels(data.models)).catch(() => setModels({}));
    void refresh(); const timer = setInterval(refresh, 10000); return () => clearInterval(timer);
  }, []);
  async function findAddress() {
    if (address.trim().length < 3) return;
    setSearching(true); setError('');
    try {
      const matches = await request<typeof searchResults>(`/api/search?q=${encodeURIComponent(address)}`);
      setSearchResults(matches); if (!matches.length) setError('No address found. Try a street and town, or click on the map.');
    } catch (e) { setError((e as Error).message); } finally { setSearching(false); }
  }
  return <main className="workspace">
    <aside className="panel">
      <header><div className="eyebrow">SWISS ROOF EXPLORER</div><h1>PV Visual Placer</h1><p>Choose a house. See what fits.</p></header>
      <form onSubmit={e => { e.preventDefault(); void findAddress(); }} className="search-form">
        <label htmlFor="address">Find an address</label><div className="search-row"><Input id="address" value={address} onChange={e => setAddress(e.target.value)} placeholder="Street, number, town" /><Button type="submit" disabled={searching}>{searching ? '…' : 'Find'}</Button></div>
      </form>
      {!!searchResults.length && <div className="search-results">{searchResults.map((r, i) => <Button variant="ghost" key={i} onClick={() => { map.current?.setView([r.lat, r.lon], 20); setSearchResults([]); setMessage('Click inside the roof you want to analyze.'); }}>{r.label.replace(/<[^>]+>/g, '')}</Button>)}</div>}
      <div className="model-status">{Object.entries(models).map(([name, m]) => <div key={name}><span className={m.available ? 'dot ready' : 'dot'} />{name === 'swiss' ? 'Swiss PV' : name === 'roof' ? 'Roof boundaries' : 'Roof obstacles'}<strong>{m.state === 'training' ? `Training ${m.epoch ?? 0}/${m.epochs_requested}` : m.available ? 'Ready' : m.state.replaceAll('_', ' ')}</strong></div>)}{!Object.keys(models).length && <span>Connecting to local model server…</span>}</div>
      <div className="legend"><span><i className="roof" />Roof</span><span><i className="pv" />Existing PV</span><span><i className="obstacle" />Obstacles</span><span><i className="new" />New panels</span></div>
      <output className="selection-status">{busy && <span className="spinner" />}{message}</output>
      {error && <div className="error" role="alert">{error}</div>}
      {result?.provisional && <p className="error">Provisional layout: obstacle or roof-boundary models are still training or unavailable. Do not treat empty predictions as a clear roof.</p>}
      {result && <><div className="metrics"><div><strong>{result.panel_count}</strong><span>new panels</span></div><div><strong>{result.additional_kwp.toFixed(2)}</strong><span>additional kWp</span></div><div><strong>{result.annual_kwh?.toLocaleString() ?? '—'}</strong><span>estimated kWh/year</span></div><div><strong>{result.usable_area_m2}</strong><span>usable roof m²</span></div></div>
        <p className="small">Detected existing PV footprint: {result.existing_pv_area_m2} m²</p>
        <table><caption>Roof faces</caption><thead><tr><th>Face</th><th>Pitch</th><th>Direction</th><th>Panels</th></tr></thead><tbody>{result.facets.map((f,i) => <tr key={f.id}><td>{i+1}</td><td>{f.pitch_deg.toFixed(0)}°</td><td>{f.azimuth_deg.toFixed(0)}°</td><td>{f.panel_count}</td></tr>)}</tbody></table></>}
      <div className="module-info"><strong>Trina Vertex S+ · 450 W</strong><span>1.762 × 1.134 m · portrait or landscape</span><a href="https://www.trinasolar.com/en-glb/NEG9RC.27/" target="_blank" rel="noreferrer">Manufacturer dimensions ↗</a></div>
      <div className="settings"><label htmlFor="setback">Roof edge clearance (m)<Input id="setback" type="number" min="0.3" max="2" step="0.1" value={setback} onChange={e => { const v=Number(e.target.value); if(Number.isFinite(v)) setSetback(Math.min(2,Math.max(.3,v))); }} /></label><label htmlFor="confidence">PV / roof confidence<Input id="confidence" type="number" min="0.1" max="0.9" step="0.05" value={confidence} onChange={e => { const v=Number(e.target.value); if(Number.isFinite(v)) setConfidence(Math.min(.9,Math.max(.1,v))); }} /></label><label htmlFor="obstacle-confidence">Obstacle confidence<Input id="obstacle-confidence" type="number" min="0.1" max="0.9" step="0.05" value={obstacleConfidence} onChange={e => { const v=Number(e.target.value); if(Number.isFinite(v)) setObstacleConfidence(Math.min(.9,Math.max(.1,v))); }} /></label><label htmlFor="row-gap">Space between rows (m)<Input id="row-gap" type="number" min="0.2" max="2" step="0.05" value={rowGap} onChange={e => { const v=Number(e.target.value);if(Number.isFinite(v))setRowGap(Math.min(2,Math.max(.2,v))); }} /></label></div>
      <p className="small">10 cm module gaps · at least 1 m between flat-roof rows · 80 cm access corridor on large faces.</p>
      <Button className="reanalyze" disabled={!selected || busy} onClick={() => selected && analyze(...selected)}>Analyze selected house again</Button>
      {result && <details><summary>Assumptions and limits</summary><ul>{result.warnings.map(w => <li key={w}>{w}</li>)}</ul></details>}
      <footer>Prototype layout · review detections before using results.<br />Direction: 0° north, 90° east, 180° south.</footer>
    </aside>
    <section className="map-wrap" aria-label="Interactive Swiss aerial map"><div ref={mapElement} className="map" /><div className="map-instruction">{busy ? 'Analyzing roof…' : 'Click a roof to place panels'}</div></section>
  </main>;
}
