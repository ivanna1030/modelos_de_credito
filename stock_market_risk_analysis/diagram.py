import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

fig, ax = plt.subplots(figsize=(16, 11))
ax.set_xlim(0, 16)
ax.set_ylim(0, 11)
ax.axis('off')
fig.patch.set_facecolor('#ffffff')
ax.set_facecolor('#ffffff')

# ── Paleta de colores ─────────────────────────────────────────────────────────
C_DARK  = '#5a1e35'
C_MID   = '#92425e'
C_SOFT  = '#b06080'
C_LIGHT = '#d4a0b5'

# ── Funciones helper ──────────────────────────────────────────────────────────

def box(ax, x, y, w, h, label, sublabel='',
        bg='#f5e0ea', border='#92425e', fontsize=8.5, bold=False):
    """
    Dibuja un rectángulo redondeado con etiqueta y sublabel opcional.
      x, y    = centro del rectángulo
      w, h    = ancho y alto
      label   = texto principal
      sublabel= texto secundario (más pequeño, debajo)
    """
    rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                          boxstyle="round,pad=0.08",
                          facecolor=bg, edgecolor=border, linewidth=1.8, zorder=3)
    ax.add_patch(rect)
    fw = 'bold' if bold else 'normal'
    ax.text(x, y + (0.13 if sublabel else 0), label,
            ha='center', va='center', color=C_DARK,
            fontsize=fontsize, fontweight=fw, zorder=4, family='monospace')
    if sublabel:
        ax.text(x, y - 0.22, sublabel, ha='center', va='center',
                color=C_SOFT, fontsize=6.5, zorder=4, family='monospace')

def arrow(ax, x1, y1, x2, y2, color='#92425e'):
    """Dibuja una flecha de (x1,y1) a (x2,y2)."""
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.5), zorder=2)

def file_tag(ax, x, y, name, color='#92425e'):
    """Dibuja una etiqueta pequeña con el nombre del archivo."""
    ax.text(x, y, name, ha='center', va='center',
            color=color, fontsize=6.5, family='monospace',
            bbox=dict(boxstyle='round,pad=0.2', facecolor='#ffffff',
                      edgecolor=color, linewidth=0.8))

# ── Título ────────────────────────────────────────────────────────────────────
ax.text(8, 10.55, 'FUNCTIONAL DIAGRAM', ha='center', va='center',
        color=C_DARK, fontsize=15, fontweight='bold', family='monospace')
ax.text(8, 10.15, 'Stock Market Risk Analysis  ·  Altman Z-Score + Merton Model',
        ha='center', va='center', color=C_MID, fontsize=9, family='monospace')
ax.plot([1, 15], [9.85, 9.85], color=C_LIGHT, lw=1)

# ── Fila 1: analyze_portfolio ─────────────────────────────────────────────────
box(ax, 8, 9.2, 4.5, 0.62,
    'analyze_portfolio([symbols])',
    'entry point — map() over each ticker',
    bg='#edd5e0', border=C_DARK, bold=True)
file_tag(ax, 11.2, 9.2, 'main.py', C_DARK)
arrow(ax, 8, 8.89, 8, 8.42)

# ── Fila 2: analyze_company ───────────────────────────────────────────────────
box(ax, 8, 8.1, 4.2, 0.62,
    'analyze_company(symbol)',
    'composes altman + merton + decision',
    bg='#f0d5e2', border=C_MID, bold=True)
file_tag(ax, 11.4, 8.1, 'credit_decision.py', C_MID)
arrow(ax, 8, 7.79, 8, 7.32)

# ── Fila 3: fetch_ticker ──────────────────────────────────────────────────────
# centro=(8, 7.0), fondo centro=(8, 6.69)
box(ax, 8, 7.0, 3.2, 0.62,
    'fetch_ticker(symbol)',
    'yfinance.Ticker object',
    bg='#f5e8ef', border=C_SOFT)
file_tag(ax, 10.8, 7.0, 'data_fetcher.py', C_SOFT)

# ── Flechas diagonales hacia las ramas ───────────────────────────────────────
# Origen: fondo centro de fetch_ticker (8, 6.69)
# Destino: tope centro de cada extract box (2.8, 6.11) y (13.2, 6.11)
arrow(ax, 6.4, 7, 2.45, 6.11)
arrow(ax, 9.6, 7, 13.55, 6.11)

# ── Etiquetas de rama (en el punto medio de cada flecha) ─────────────────────
# Punto medio flecha izquierda: x=(8+2.8)/2=5.4,  y=(6.69+6.11)/2=6.4
# Punto medio flecha derecha:   x=(8+13.2)/2=10.6, y=6.4
ax.text(2.795, 6.5, 'ALTMAN BRANCH', ha='center', va='center',
        color=C_MID, fontsize=7, family='monospace', style='italic')
ax.text(13.205, 6.5, 'MERTON BRANCH', ha='center', va='center',
        color=C_MID, fontsize=7, family='monospace', style='italic')

# ── Columna ALTMAN (izquierda) ────────────────────────────────────────────────
ax_l = 2.8   # centro x de la columna altman

box(ax, ax_l, 5.8, 3.4, 0.62,
    'extract_altman_vars()',
    'balance sheet + income stmt',
    bg='#f5e8ef', border=C_MID)
arrow(ax, ax_l, 5.49, ax_l, 5.02)

box(ax, ax_l, 4.7, 3.4, 0.62,
    'compute_altman_ratios()',
    'X1, X2, X3, X4, X5',
    bg='#f5e8ef', border=C_MID)
arrow(ax, ax_l, 4.39, ax_l, 3.92)

box(ax, ax_l, 3.6, 3.4, 0.62,
    'altman_zscore(ratios)',
    'Z = 1.2X1+1.4X2+3.3X3+0.6X4+X5',
    bg='#f5e8ef', border=C_MID)
arrow(ax, ax_l, 3.29, ax_l, 2.82)

box(ax, ax_l, 2.5, 3.4, 0.62,
    'classify_zscore(z)',
    'Safe  /  Grey  /  Distress',
    bg='#edd5e0', border=C_DARK)
file_tag(ax, ax_l, 1.9, 'altman.py', C_MID)

# ── Columna MERTON (derecha) ──────────────────────────────────────────────────
ax_r = 13.2  # centro x de la columna merton

box(ax, ax_r, 5.8, 3.4, 0.62,
    'extract_merton_vars()',
    'equity, debt, sigma_E, r, T',
    bg='#f5e8ef', border=C_MID)
arrow(ax, ax_r, 5.49, ax_r, 5.02)

box(ax, ax_r, 4.7, 3.4, 0.62,
    'merton_asset_value()',
    'iter. Black-Scholes → V, sigma_V',
    bg='#f5e8ef', border=C_MID)
arrow(ax, ax_r, 4.39, ax_r, 3.92)

box(ax, ax_r, 3.6, 3.4, 0.62,
    'merton_default_probability()',
    'PD = 1 - N(DD)',
    bg='#f5e8ef', border=C_MID)
arrow(ax, ax_r, 3.29, ax_r, 2.82)

box(ax, ax_r, 2.5, 3.4, 0.62,
    'classify_merton(pd)',
    'Low  /  Medium  /  High',
    bg='#edd5e0', border=C_DARK)
file_tag(ax, ax_r, 1.9, 'merton.py', C_MID)

# ── Flechas de convergencia hacia credit_decision ─────────────────────────────
arrow(ax, 4.5, 2.5, 8.3, 1.62)
arrow(ax, 11.5, 2.5, 7.7, 1.62)

# ── credit_decision ───────────────────────────────────────────────────────────
box(ax, 8, 1.3, 5.0, 0.62,
    'credit_decision(z_score, default_prob)',
    'Z > 1.8  AND  PD < 20%  →  APPROVED / DENIED',
    bg='#edd5e0', border=C_DARK, bold=True)
file_tag(ax, 11.8, 1.3, 'credit_decision.py', C_DARK)
arrow(ax, 8, 0.99, 8, 0.62)

# ── Output final ──────────────────────────────────────────────────────────────
box(ax, 8, 0.35, 3.8, 0.45,
    'APPROVED  /  DENIED',
    bg='#fdf0f4', border=C_MID, fontsize=8.5, bold=True)

# ── Nota dashboard ────────────────────────────────────────────────────────────
file_tag(ax, 8, -0.16, 'dashboard.py', C_SOFT)

ax.text(8, -0.36,
        'streamlit UI  ·  calls analyze_portfolio() and renders all charts',
        ha='center', va='center', color=C_SOFT, fontsize=7,
        family='monospace', style='italic')

# ── Guardar ───────────────────────────────────────────────────────────────────
plt.tight_layout(pad=0.5)
plt.savefig('functional_diagram.png', dpi=160,
            bbox_inches='tight', facecolor='#ffffff')
print("Saved! -> functional_diagram.png")