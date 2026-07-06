import numpy as np
import matplotlib.pyplot as plt

# Set style agar terlihat bersih dan profesional
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['axes.edgecolor'] = '#333333'
plt.rcParams['axes.linewidth'] = 0.8

# =============================================================================
# GRAFIK 1: Tiga Kondisi Fungsi Klaster
# =============================================================================

# Definisi fungsi untuk grafik 1 (Double well-like shape)
def F1(x):
    return 0.15 * x**4 - 0.2 * x**3 - 1.2 * x**2 + 0.5 * x + 1.0

x1 = np.linspace(-3.2, 3.2, 500)

fig, axs = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
fig.suptitle(r"Tiga Kondisi Fungsi Klaster $\mathcal{FK}(\mathbf{y})$  $-$  $\mathbf{x}_t = \frac{1}{2}(\mathbf{y} + \mathbf{x}_c)$", fontsize=14, fontweight='bold', y=0.98)

# --- Kondisi 1 ---
ax = axs[0]
ax.plot(x1, F1(x1), color='#2c3e50', linewidth=2)
xc, y = -1.8, 1.4
xt = 0.5 * (y + xc)
ax.scatter(xc, F1(xc), color='#2980b9', s=150, zorder=5)
ax.scatter(y, F1(y), color='#d35400', s=150, zorder=5)
ax.scatter(xt, F1(xt), color='#27ae60', marker='D', s=120, zorder=5)
ax.vlines([xc, y, xt], -1.5, [F1(xc), F1(y), F1(xt)], colors='gray', linestyles='dashed', alpha=0.7)
ax.text(xc-0.3, F1(xc)+0.1, r'$\mathbf{x}_c$', color='#2980b9', fontsize=12, fontweight='bold')
ax.text(y+0.1, F1(y)+0.1, r'$\mathbf{y}$', color='#d35400', fontsize=12, fontweight='bold')
ax.text(xt+0.1, F1(xt)-0.2, r'$\mathbf{x}_t$ (terkecil)', color='#27ae60', fontsize=11, fontweight='bold')
ax.text(0.5, 1.5, '$\u2192$ Klaster baru\ndi $\mathbf{y}$', color='#4a235a', fontsize=11, fontweight='bold', ha='center')
ax.set_title("Kondisi 1:\n" + r"$F(\mathbf{x}_t) < F(\mathbf{y})$ dan $F(\mathbf{x}_t) < F(\mathbf{x}_c)$", fontsize=11)

# --- Kondisi 2 ---
ax = axs[1]
ax.plot(x1, F1(x1), color='#2c3e50', linewidth=2)
xc, y = -1.1, 1.5
xt = 0.5 * (y + xc)
ax.scatter(xc, F1(xc), color='#2980b9', s=150, zorder=5)
ax.scatter(y, F1(y), color='#d35400', s=150, zorder=5)
ax.scatter(xt, F1(xt), color='#27ae60', marker='D', s=120, zorder=5)
ax.vlines([xc, y, xt], -1.5, [F1(xc), F1(y), F1(xt)], colors='gray', linestyles='dashed', alpha=0.7)
ax.text(xc-0.3, F1(xc)+0.1, r'$\mathbf{x}_c$', color='#2980b9', fontsize=12, fontweight='bold')
ax.text(y+0.1, F1(y)+0.1, r'$\mathbf{y}$', color='#d35400', fontsize=12, fontweight='bold')
ax.text(xt+0.1, F1(xt)-0.1, r'$\mathbf{x}_t$ (terbesar)', color='#27ae60', fontsize=11, fontweight='bold')
ax.text(-0.2, 1.8, '$\u2192$ Klaster baru di $\mathbf{y}$\n+ panggil $\mathcal{FK}(\mathbf{x}_t)$', color='#4a235a', fontsize=11, fontweight='bold', ha='center')
ax.set_title("Kondisi 2:\n" + r"$F(\mathbf{x}_t) > F(\mathbf{y})$ dan $F(\mathbf{x}_t) > F(\mathbf{x}_c)$", fontsize=11)

# --- Kondisi 3 (SUDAH DITUKAR) ---
ax = axs[2]
ax.plot(x1, F1(x1), color='#2c3e50', linewidth=2)
# Tukar posisi: xc di kiri (-0.5), y di kanan (2.3)
xc, y = -0.5, 2.3
xt = 0.5 * (y + xc)
ax.scatter(xc, F1(xc), color='#2980b9', s=150, zorder=5) 
ax.scatter(y, F1(y), color='#d35400', s=150, zorder=5)  
ax.scatter(xt, F1(xt), color='#27ae60', marker='D', s=120, zorder=5)
ax.vlines([xc, y, xt], -1.5, [F1(xc), F1(y), F1(xt)], colors='gray', linestyles='dashed', alpha=0.7)
ax.text(xc-0.1, F1(xc)-0.3, r'$\mathbf{x}_c$ (dipertahankan)', color='#2980b9', fontsize=11, fontweight='bold', ha='center')
ax.text(y+0.1, F1(y), r'$\mathbf{y}$ (diabaikan)', color='#d35400', fontsize=11, fontweight='bold')
ax.text(xt+0.1, F1(xt)+0.1, r'$\mathbf{x}_t$', color='#27ae60', fontsize=12, fontweight='bold')
ax.set_title("Kondisi 3:\n" + r"$F(\mathbf{y}) < F(\mathbf{x}_c)$ $-$ $\mathbf{y}$ diabaikan", fontsize=11)

# Formatting estetika untuk Grafik 1
for ax in axs:
    ax.set_xlabel('$X$', fontsize=11)
    ax.set_ylabel('$F(X)$', fontsize=11)
    ax.set_xlim(-3.2, 3.2)
    ax.set_ylim(-1.2, 4.2)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_facecolor('#fdfefe')

# Legend bawah untuk Grafik 1
labels = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980b9', markersize=12, label=r'$\mathbf{x}_c$ - pusat klaster terdekat'),
          plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d35400', markersize=12, label=r'$\mathbf{y}$ - titik kandidat'),
          plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#27ae60', markersize=10, label=r'$\mathbf{x}_t$ - titik tengah')]
fig.legend(handles=labels, loc='lower center', ncol=3, bbox_to_anchor=(0.5, -0.05), fontsize=11)
plt.tight_layout()
plt.savefig('visualisasi_fungsi_klaster_fixed.png', dpi=300, bbox_inches='tight')


# =============================================================================
# GRAFIK 2: Pengaruh Parameter np terhadap Titik Pemeriksaan (SUDAH DITUKAR)
# =============================================================================

# Definisi fungsi untuk grafik 2 (Wavy upward trend)
def F2(x):
    return 0.5 * x + 0.4 * np.sin(3.5 * x) + 0.2

x2 = np.linspace(-0.3, 5.0, 500)

fig2, axs2 = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
fig2.suptitle(r"Pengaruh Parameter $n_p$ terhadap Titik Pemeriksaan pada Fungsi Klaster $\mathcal{FK}$", fontsize=14, fontweight='bold', y=0.98)

# Tukar posisi untuk kedua grafik: y di kiri (0.7), xc di kanan (4.0)
y_pos, xc_pos = 0.7, 4.0

# --- Subplot 1: np = 1 ---
ax = axs2[0]
ax.plot(x2, F2(x2), color='#2c3e50', linewidth=2.5)
ax.scatter(y_pos, F2(y_pos), color='#d35400', s=150, zorder=5)
ax.scatter(xc_pos, F2(xc_pos), color='#2980b9', s=150, zorder=5)
# np = 1 -> Hanya ada 1 titik tengah xt1
xt1 = 0.5 * (y_pos + xc_pos)
ax.scatter(xt1, F2(xt1), color='#e74c3c', marker='X', s=200, zorder=6)
ax.vlines([y_pos, xc_pos, xt1], -0.6, [F2(y_pos), F2(xc_pos), F2(xt1)], colors=['#d35400', '#2980b9', '#e74c3c'], linestyles='dashed', alpha=0.8, linewidth=1.5)
ax.text(y_pos+0.1, F2(y_pos)+0.05, r'$\mathbf{y}$', color='#d35400', fontsize=13, fontweight='bold')
ax.text(xc_pos+0.05, F2(xc_pos)+0.05, r'$\mathbf{x}_C$', color='#2980b9', fontsize=13, fontweight='bold')
ax.text(xt1+0.05, F2(xt1)+0.05, r'$\mathbf{x}_{t_1}$', color='#e74c3c', fontsize=11)
ax.set_title(r"$n_p = 1$" + "\n" + r"$\times$ Tidak ada titik yang memenuhi kondisi FK", fontsize=13, color='maroon', fontweight='bold')

# --- Subplot 2: np = 3 ---
ax = axs2[1]
ax.plot(x2, F2(x2), color='#2c3e50', linewidth=2.5)
ax.scatter(y_pos, F2(y_pos), color='#d35400', s=150, zorder=5)
ax.scatter(xc_pos, F2(xc_pos), color='#2980b9', s=150, zorder=5)
# np = 3 -> ada beberapa titik uji (misal simulasi pembagian interval)
xt_ticks = [1.5, 2.35, 3.2]
colors_ticks = ['#27ae60', '#e74c3c', '#27ae60']
markers_ticks = ['D', 'X', 'D']
for i, (xt, col, mark) in enumerate(zip(xt_ticks, colors_ticks, markers_ticks)):
    size = 120 if mark == 'D' else 200
    ax.scatter(xt, F2(xt), color=col, marker=mark, s=size, zorder=6)
    ax.vlines(xt, -0.6, F2(xt), colors=col, linestyles='dashed', alpha=0.8, linewidth=1.5)
    ax.text(xt+0.05, F2(xt)+0.08, r'$\mathbf{x}_{t_' + str(i+1) + '}$', color=col, fontsize=11)
ax.vlines([y_pos, xc_pos], -0.6, [F2(y_pos), F2(xc_pos)], colors=['#d35400', '#2980b9'], linestyles='dashed', alpha=0.8, linewidth=1.5)
ax.text(y_pos+0.1, F2(y_pos)+0.05, r'$\mathbf{y}$', color='#d35400', fontsize=13, fontweight='bold')
ax.text(xc_pos+0.05, F2(xc_pos)+0.05, r'$\mathbf{x}_C$', color='#2980b9', fontsize=13, fontweight='bold')
ax.set_title(r"$n_p = 3$" + "\n" + r"$\checkmark$ Ada titik yang memenuhi kondisi FK", fontsize=13, color='green', fontweight='bold')

# Formatting estetika untuk Grafik 2
for ax in axs2:
    ax.set_xlabel('$x$', fontsize=12)
    ax.set_ylabel('$F(x)$', fontsize=12)
    ax.set_xlim(-0.3, 5.0)
    ax.set_ylim(-0.6, 3.4)
    ax.grid(True, linestyle=':', alpha=0.5)
    ax.set_facecolor('#fdfefe')

# Legend bawah untuk Grafik 2
labels2 = [plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#2980b9', markersize=12, label=r'$\mathbf{x}_C$ — pusat klaster'),
           plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='#d35400', markersize=12, label=r'$\mathbf{y}$ — titik kandidat'),
           plt.Line2D([0], [0], marker='D', color='w', markerfacecolor='#27ae60', markersize=10, label=r'Titik $n_p$  $\checkmark$ memenuhi kondisi FK'),
           plt.Line2D([0], [0], marker='X', color='w', markerfacecolor='#e74c3c', markersize=12, label=r'Titik $n_p$  $\times$ tidak memenuhi kondisi FK')]
fig2.legend(handles=labels2, loc='lower center', ncol=4, bbox_to_anchor=(0.5, -0.05), fontsize=10.5)
plt.tight_layout()
plt.savefig('visualisasi_np_titik_tengah_fixed.png', dpi=300, bbox_inches='tight')

plt.show()