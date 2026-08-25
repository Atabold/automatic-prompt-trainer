"""Shared Gradio theme and stylesheet."""
import gradio as gr

APP_TITLE = "Prompt Trainer"
APP_SUBTITLE = "Local GGUF formatter · multi-provider evaluation · prompt optimization"


def build_theme() -> gr.Theme:
    return gr.themes.Soft(
        primary_hue=gr.themes.colors.indigo,
        secondary_hue=gr.themes.colors.slate,
        neutral_hue=gr.themes.colors.slate,
        radius_size=gr.themes.sizes.radius_sm,
        font=(gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"),
        font_mono=(gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"),
    ).set(
        body_background_fill="*neutral_50",
        block_background_fill="white",
        block_border_width="1px",
        block_shadow="0 1px 2px rgba(15, 23, 42, 0.05)",
        block_label_text_weight="600",
        block_title_text_weight="600",
        button_primary_shadow="none",
        button_secondary_shadow="none",
    )


CSS = """
:root {
  --pt-ok: #059669;
  --pt-warn: #d97706;
  --pt-bad: #dc2626;
  --pt-muted: #64748b;
  --pt-line: rgba(100, 116, 139, 0.22);
  --pt-surface: rgba(100, 116, 139, 0.06);
}
.dark {
  --pt-ok: #34d399;
  --pt-warn: #fbbf24;
  --pt-bad: #f87171;
  --pt-muted: #94a3b8;
  --pt-line: rgba(148, 163, 184, 0.25);
  --pt-surface: rgba(148, 163, 184, 0.10);
}

.gradio-container { max-width: 1400px !important; }

/* ---------- header ---------- */
#pt-header {
  display: flex; align-items: center; justify-content: space-between;
  gap: 16px; flex-wrap: wrap;
  padding: 18px 22px; margin-bottom: 6px;
  border: 1px solid var(--pt-line); border-radius: 10px;
  background: linear-gradient(100deg, rgba(99,102,241,0.10), rgba(99,102,241,0.02));
}
#pt-header .pt-brand { display: flex; align-items: baseline; gap: 12px; flex-wrap: wrap; }
#pt-header h1 { margin: 0; font-size: 1.3rem; font-weight: 700; letter-spacing: -0.01em; }
#pt-header .pt-sub { margin: 0; color: var(--pt-muted); font-size: 0.82rem; }

/* ---------- pills ---------- */
.pt-pills { display: flex; gap: 8px; flex-wrap: wrap; }
.pt-pill {
  display: inline-flex; align-items: center; gap: 6px;
  padding: 4px 11px; border-radius: 999px;
  border: 1px solid var(--pt-line); background: var(--pt-surface);
  font-size: 0.76rem; font-weight: 500; white-space: nowrap;
}
.pt-pill b { font-weight: 600; }
.pt-pill.ok    { border-color: color-mix(in srgb, var(--pt-ok) 45%, transparent); }
.pt-pill.warn  { border-color: color-mix(in srgb, var(--pt-warn) 55%, transparent); }
.pt-pill.bad   { border-color: color-mix(in srgb, var(--pt-bad) 45%, transparent); }
.pt-dot { width: 7px; height: 7px; border-radius: 50%; background: var(--pt-muted); }
.pt-pill.ok .pt-dot   { background: var(--pt-ok); }
.pt-pill.warn .pt-dot { background: var(--pt-warn); }
.pt-pill.bad .pt-dot  { background: var(--pt-bad); }

/* ---------- stat tiles ---------- */
.pt-stats { display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr)); gap: 10px; }
.pt-stat {
  padding: 12px 14px; border: 1px solid var(--pt-line);
  border-radius: 9px; background: var(--pt-surface);
}
.pt-stat .pt-stat-label {
  font-size: 0.7rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--pt-muted); font-weight: 600;
}
.pt-stat .pt-stat-value { font-size: 1.45rem; font-weight: 700; line-height: 1.25; margin-top: 3px; }
.pt-stat .pt-stat-note { font-size: 0.74rem; color: var(--pt-muted); }
.pt-stat.ok .pt-stat-value  { color: var(--pt-ok); }
.pt-stat.bad .pt-stat-value { color: var(--pt-bad); }

/* ---------- status banner ---------- */
.pt-status {
  display: flex; align-items: center; gap: 11px;
  padding: 11px 14px; border-radius: 9px;
  border: 1px solid var(--pt-line); background: var(--pt-surface);
  font-size: 0.87rem;
}
.pt-status.running { border-left: 3px solid #6366f1; }
.pt-status.ok      { border-left: 3px solid var(--pt-ok); }
.pt-status.warn    { border-left: 3px solid var(--pt-warn); }
.pt-status.bad     { border-left: 3px solid var(--pt-bad); }
.pt-status .pt-status-title { font-weight: 600; }
.pt-status .pt-status-detail { color: var(--pt-muted); }
.pt-spinner {
  width: 13px; height: 13px; flex: none; border-radius: 50%;
  border: 2px solid var(--pt-line); border-top-color: #6366f1;
  animation: pt-spin 0.8s linear infinite;
}
@keyframes pt-spin { to { transform: rotate(360deg); } }

/* ---------- progress bar ---------- */
.pt-bar { height: 6px; border-radius: 999px; background: var(--pt-surface); overflow: hidden; margin-top: 9px; }
.pt-bar > span { display: block; height: 100%; background: #6366f1; transition: width 0.25s ease; }

/* ---------- test result cards ---------- */
.pt-cards { display: flex; flex-direction: column; gap: 9px; max-height: 620px; overflow-y: auto; padding-right: 4px; }
.pt-card { border: 1px solid var(--pt-line); border-radius: 9px; overflow: hidden; background: var(--pt-surface); }
.pt-card > summary {
  display: flex; align-items: center; gap: 10px; flex-wrap: wrap;
  padding: 9px 13px; cursor: pointer; list-style: none; font-size: 0.85rem;
}
.pt-card > summary::-webkit-details-marker { display: none; }
.pt-card.pass { border-left: 3px solid var(--pt-ok); }
.pt-card.fail { border-left: 3px solid var(--pt-bad); }
.pt-badge {
  font-size: 0.68rem; font-weight: 700; letter-spacing: 0.05em;
  padding: 2px 8px; border-radius: 4px; text-transform: uppercase;
}
.pt-badge.pass { background: color-mix(in srgb, var(--pt-ok) 18%, transparent); color: var(--pt-ok); }
.pt-badge.fail { background: color-mix(in srgb, var(--pt-bad) 18%, transparent); color: var(--pt-bad); }
.pt-card .pt-id { font-family: var(--font-mono); font-weight: 600; }
.pt-card .pt-score { margin-left: auto; font-variant-numeric: tabular-nums; font-weight: 600; }
.pt-card .pt-preview { color: var(--pt-muted); overflow: hidden; text-overflow: ellipsis; white-space: nowrap; max-width: 45%; }
.pt-card .pt-body { padding: 4px 13px 13px; display: grid; gap: 9px; }
.pt-field { display: grid; gap: 3px; }
.pt-field .pt-field-label {
  font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.06em;
  color: var(--pt-muted); font-weight: 600;
}
.pt-field pre {
  margin: 0; padding: 8px 10px; border-radius: 6px;
  background: var(--pt-surface); border: 1px solid var(--pt-line);
  font-size: 0.8rem; white-space: pre-wrap; word-break: break-word;
  font-family: var(--font-mono);
}
.pt-field.out pre { border-left: 2px solid #6366f1; }
.pt-subscores { display: flex; gap: 7px; flex-wrap: wrap; font-size: 0.72rem; color: var(--pt-muted); }
.pt-subscores span { padding: 2px 8px; border: 1px solid var(--pt-line); border-radius: 999px; }

/* ---------- generic table ---------- */
.pt-table { width: 100%; border-collapse: collapse; font-size: 0.83rem; }
.pt-table th {
  text-align: left; padding: 7px 11px; font-size: 0.7rem; text-transform: uppercase;
  letter-spacing: 0.06em; color: var(--pt-muted); border-bottom: 1px solid var(--pt-line);
}
.pt-table td { padding: 8px 11px; border-bottom: 1px solid var(--pt-line); vertical-align: top; }
.pt-table tr:last-child td { border-bottom: none; }
.pt-table td.num { text-align: right; font-variant-numeric: tabular-nums; }
.pt-table tr.best td { background: color-mix(in srgb, var(--pt-ok) 10%, transparent); }
.pt-mono { font-family: var(--font-mono); font-size: 0.8rem; }

.pt-empty {
  padding: 26px; text-align: center; color: var(--pt-muted); font-size: 0.87rem;
  border: 1px dashed var(--pt-line); border-radius: 9px;
}
.pt-hint { color: var(--pt-muted); font-size: 0.8rem; }
footer { display: none !important; }
"""
