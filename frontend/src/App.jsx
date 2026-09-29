import { useState, useCallback } from 'react';

// ── Config ──────────────────────────────────────────────────────────────────
// Empty string = relative URLs, proxied by Vite to localhost:8000 in dev.
// Set to 'http://localhost:8000' if serving the built dist directly.
const API_BASE = '';

const SCENARIOS = [
  {
    label: 'MySQL Connection Failure',
    issue: 'Payment API cannot connect to MySQL',
    deployment: { status: 'unhealthy' },
  },
  {
    label: 'Service Down',
    issue: 'Health check failing, service not responding',
    deployment: { status: 'unhealthy' },
  },
  {
    label: 'High CPU',
    issue: 'High CPU usage spike in the payment service',
    deployment: { status: 'degraded' },
  },
  {
    label: 'Crashloop',
    issue: 'Container is crash looping and restarting repeatedly',
    deployment: { status: 'unhealthy' },
  },
  {
    label: 'High Memory',
    issue: 'Application running out of memory, heap OOM killed',
    deployment: { status: 'unhealthy' },
  },
];

// ── Small UI primitives ──────────────────────────────────────────────────────

function Badge({ severity }) {
  const map = {
    critical: 'badge-critical',
    high:     'badge-high',
    medium:   'badge-medium',
    low:      'badge-low',
  };
  return (
    <span className={`badge ${map[severity] ?? 'badge-medium'}`}>
      {severity?.toUpperCase() ?? '—'}
    </span>
  );
}

function SourceTag({ source }) {
  const map = {
    chromadb: 'tag-chroma',
    keyword:  'tag-keyword',
    ollama:   'tag-ollama',
    rules:    'tag-rules',
    none:     'tag-none',
  };
  const labels = {
    chromadb: 'ChromaDB',
    keyword:  'Keyword',
    ollama:   'Ollama LLM',
    rules:    'Rule-based',
    none:     'None',
  };
  return (
    <span className={`source-tag ${map[source] ?? 'tag-none'}`}>
      {labels[source] ?? source}
    </span>
  );
}

function ConfidenceBar({ value }) {
  const pct = Math.round((value ?? 0) * 100);
  const cls = pct >= 80 ? 'conf-high' : pct >= 50 ? 'conf-mid' : 'conf-low';
  return (
    <div className="conf-wrap">
      <div className={`conf-bar ${cls}`} style={{ width: `${pct}%` }} />
      <span className="conf-label">{pct}%</span>
    </div>
  );
}

function StepEvent({ event }) {
  const icons = {
    start:        '🚀',
    diagnosis:    '🔍',
    knowledge:    '📚',
    remediation:  '🔧',
    verification: '✅',
    complete:     '🎯',
    error:        '❌',
  };
  return (
    <div className={`step-event ${event.event === 'error' ? 'step-error' : ''}`}>
      <span className="step-icon">{icons[event.event] ?? '•'}</span>
      <span className="step-name">{event.event}</span>
      {event.root_cause    && <span className="step-detail">{event.root_cause}</span>}
      {event.severity      && <Badge severity={event.severity} />}
      {event.runbook_title && <span className="step-detail">{event.runbook_title}</span>}
      {event.action        && <span className="step-detail">{event.action}</span>}
      {event.status        && <span className="step-detail">status: {event.status}</span>}
      {event.detail        && <span className="step-detail step-error-msg">{event.detail}</span>}
    </div>
  );
}

function TrailStep({ step, index }) {
  return (
    <div className="trail-step">
      <span className="trail-idx">{index + 1}</span>
      <span className="trail-agent">{step.next_agent}</span>
      <span className="trail-reason">{step.reason}</span>
    </div>
  );
}

// ── Main App ─────────────────────────────────────────────────────────────────

export default function App() {
  const [scenario, setScenario]     = useState(0);
  const [dryRun, setDryRun]         = useState(true);
  const [mode, setMode]             = useState('stream'); // 'stream' | 'sync'
  const [loading, setLoading]       = useState(false);
  const [streamEvents, setStreamEvents] = useState([]);
  const [result, setResult]         = useState(null);
  const [error, setError]           = useState('');

  const currentScenario = SCENARIOS[scenario];

  // ── Stream mode ────────────────────────────────────────────────────────────
  const runStream = useCallback(async () => {
    setLoading(true);
    setError('');
    setStreamEvents([]);
    setResult(null);

    const body = JSON.stringify({
      application: 'Payment API',
      issue: currentScenario.issue,
      deployment: currentScenario.deployment,
      dry_run: dryRun,
    });

    try {
      const resp = await fetch(`${API_BASE}/workflow/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      });

      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);

      const reader = resp.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop();
        for (const line of lines) {
          if (!line.trim()) continue;
          try {
            const evt = JSON.parse(line);
            setStreamEvents(prev => [...prev, evt]);
            if (evt.event === 'complete') setResult(evt);
          } catch { /* ignore malformed lines */ }
        }
      }
    } catch (err) {
      setError(err.message ?? 'Stream error');
    } finally {
      setLoading(false);
    }
  }, [currentScenario, dryRun]);

  // ── Sync mode ──────────────────────────────────────────────────────────────
  const runSync = useCallback(async () => {
    setLoading(true);
    setError('');
    setStreamEvents([]);
    setResult(null);

    const body = JSON.stringify({
      application: 'Payment API',
      issue: currentScenario.issue,
      deployment: currentScenario.deployment,
      dry_run: dryRun,
    });

    try {
      const resp = await fetch(`${API_BASE}/workflow/run`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body,
      });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      const data = await resp.json();
      setResult(data);
    } catch (err) {
      setError(err.message ?? 'Request error');
    } finally {
      setLoading(false);
    }
  }, [currentScenario, dryRun]);

  const handleRun = mode === 'stream' ? runStream : runSync;

  // ── Derived display values ─────────────────────────────────────────────────
  const diag  = result?.diagnosis   ?? {};
  const rb    = result?.runbook     ?? {};
  const rem   = result?.remediation ?? {};
  const ver   = result?.verification ?? {};
  const trail = result?.workflow_trail ?? [];

  const resolvedStatus = result
    ? (result.resolved ? 'resolved' : 'unresolved')
    : 'idle';

  // ── Render ─────────────────────────────────────────────────────────────────
  return (
    <div className="page-shell">

      {/* ── Header ─────────────────────────────────────────────────── */}
      <header className="topbar">
        <div className="topbar-left">
          <p className="eyebrow">Local AI Operations</p>
          <h1>DeepShield AI</h1>
        </div>
        <div className="topbar-controls">
          <div className="control-row">
            <label className="control-label">Scenario</label>
            <select
              className="select-input"
              value={scenario}
              onChange={e => setScenario(Number(e.target.value))}
              disabled={loading}
            >
              {SCENARIOS.map((s, i) => (
                <option key={i} value={i}>{s.label}</option>
              ))}
            </select>
          </div>
          <div className="control-row">
            <label className="control-label">Mode</label>
            <div className="toggle-group">
              <button
                className={`toggle-btn ${mode === 'stream' ? 'active' : ''}`}
                onClick={() => setMode('stream')}
                disabled={loading}
              >Stream</button>
              <button
                className={`toggle-btn ${mode === 'sync' ? 'active' : ''}`}
                onClick={() => setMode('sync')}
                disabled={loading}
              >Sync</button>
            </div>
          </div>
          <div className="control-row">
            <label className="control-label dry-run-label">
              <input
                type="checkbox"
                checked={dryRun}
                onChange={e => setDryRun(e.target.checked)}
                disabled={loading}
              />
              Dry run
            </label>
          </div>
          <button
            className="primary-button"
            onClick={handleRun}
            disabled={loading}
          >
            {loading ? 'Running…' : 'Run self-healing cycle'}
          </button>
        </div>
      </header>

      {/* ── Hero status bar ────────────────────────────────────────── */}
      <section className="panel hero-panel">
        <div className="hero-left">
          <p className="label">Application</p>
          <h2>Payment API</h2>
          <p className="hero-issue">{currentScenario.issue}</p>
        </div>
        <div className="hero-right">
          <div className={`status-pill status-${resolvedStatus}`}>
            {resolvedStatus === 'idle'       && 'IDLE'}
            {resolvedStatus === 'resolved'   && '✓ RESOLVED'}
            {resolvedStatus === 'unresolved' && '✗ UNRESOLVED'}
          </div>
          {result && (
            <p className="exec-mode">
              via <strong>{result.execution_mode ?? 'loop'}</strong>
              {dryRun && <span className="dry-tag"> · dry run</span>}
            </p>
          )}
        </div>
      </section>

      {error && <div className="panel alert-panel">{error}</div>}

      {/* ── Stream timeline ────────────────────────────────────────── */}
      {streamEvents.length > 0 && (
        <section className="panel">
          <p className="label">Live pipeline</p>
          <div className="stream-timeline">
            {streamEvents
              .filter(e => e.event !== 'complete')
              .map((e, i) => <StepEvent key={i} event={e} />)}
          </div>
        </section>
      )}

      {/* ── Results grid ───────────────────────────────────────────── */}
      {result && (
        <>
          {/* Row 1: Diagnosis + Runbook */}
          <div className="grid two-columns">
            <article className="panel">
              <p className="label">Diagnosis</p>
              <div className="card-row">
                <SourceTag source={diag.diagnosis_source} />
                <Badge severity={diag.severity} />
              </div>
              <p className="root-cause">{diag.root_cause ?? '—'}</p>
              <p className="sublabel">Confidence</p>
              <ConfidenceBar value={diag.confidence} />
              {diag.contributing_factors?.length > 0 && (
                <>
                  <p className="sublabel" style={{ marginTop: 16 }}>Contributing factors</p>
                  <ul className="factor-list">
                    {diag.contributing_factors.map((f, i) => <li key={i}>{f}</li>)}
                  </ul>
                </>
              )}
            </article>

            <article className="panel">
              <p className="label">Knowledge base</p>
              <div className="card-row">
                <SourceTag source={rb.retrieval_source} />
                {rb.score > 0 && (
                  <span className="score-tag">score {(rb.score * 100).toFixed(0)}%</span>
                )}
              </div>
              <p className="runbook-title">{rb.title ?? '—'}</p>
              <p className="runbook-summary">{rb.summary ?? ''}</p>
              {rb.recommended_actions?.length > 0 && (
                <>
                  <p className="sublabel">Recommended actions</p>
                  <ol className="action-list">
                    {rb.recommended_actions.map((a, i) => <li key={i}>{a}</li>)}
                  </ol>
                </>
              )}
            </article>
          </div>

          {/* Row 2: Remediation + Verification */}
          <div className="grid two-columns">
            <article className="panel">
              <p className="label">Remediation</p>
              <div className="card-row">
                <span className={`risk-tag risk-${rem.risk}`}>{rem.risk?.toUpperCase()}</span>
                {rem.mutating && <span className="mutating-tag">mutating</span>}
                {rem.approved && <span className="approved-tag">approved</span>}
              </div>
              <p className="action-headline">{rem.action ?? '—'}</p>
              <p className="action-detail">{rem.details ?? ''}</p>
              {rem.action_sequence?.length > 0 && (
                <>
                  <p className="sublabel">Execution sequence</p>
                  <ol className="seq-list">
                    {rem.action_sequence.map((s, i) => <li key={i}>{s}</li>)}
                  </ol>
                </>
              )}
              <p className="sublabel" style={{ marginTop: 12 }}>
                Executed: <strong>{result.action_executed ? 'yes' : 'no (dry run)'}</strong>
              </p>
            </article>

            <article className="panel">
              <p className="label">Verification</p>
              <div className={`ver-status ver-${ver.status}`}>
                {ver.status === 'resolved' ? '✓ Service recovered' : '✗ Issues detected'}
              </div>
              <p className="sublabel">Health endpoint</p>
              <p className={`health-val health-${ver.service_health}`}>
                {ver.service_health?.toUpperCase() ?? '—'}
              </p>
              {Object.keys(ver.container_statuses ?? {}).length > 0 && (
                <>
                  <p className="sublabel">Container statuses</p>
                  <div className="container-grid">
                    {Object.entries(ver.container_statuses).map(([name, status]) => (
                      <div key={name} className="container-row">
                        <span className="container-name">{name}</span>
                        <span className={`container-status cs-${status}`}>{status}</span>
                      </div>
                    ))}
                  </div>
                </>
              )}
              {ver.issues?.length > 0 && (
                <>
                  <p className="sublabel">Issues</p>
                  <ul className="issue-list">
                    {ver.issues.map((iss, i) => <li key={i}>{iss}</li>)}
                  </ul>
                </>
              )}
            </article>
          </div>

          {/* Row 3: Supervisor trail */}
          {trail.length > 0 && (
            <section className="panel">
              <p className="label">
                Supervisor trail
                <span className="trail-meta"> · {trail.length} step(s) · {result.supervisor_status}</span>
              </p>
              <div className="trail-list">
                {trail.map((step, i) => <TrailStep key={i} step={step} index={i} />)}
              </div>
            </section>
          )}
        </>
      )}

      {/* ── Empty state ─────────────────────────────────────────────── */}
      {!result && !loading && streamEvents.length === 0 && (
        <div className="empty-state">
          <p className="empty-icon">🛡️</p>
          <p>Select a scenario and click <strong>Run self-healing cycle</strong> to start.</p>
        </div>
      )}

    </div>
  );
}
