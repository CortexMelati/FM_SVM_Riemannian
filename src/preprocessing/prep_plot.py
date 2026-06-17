"""
=============================================================================
EEG QUALITY CONTROL VISUALIZATION (Li et al., 2026 aligned)
=============================================================================
name: preprocessing_plotting.py

Objective:
    Generate a composite dashboard for visual quality control of EEG data.
    This function is called by the preprocessing pipeline to create PDF reports.
=============================================================================
"""

import numpy as np
import mne
from mne.time_frequency import tfr_multitaper
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
import sys
from pathlib import Path

# ==========================================
# 0. CONFIG IMPORT
# ==========================================
current_dir = Path(__file__).resolve().parent
sys.path.append(str(current_dir.parent))

# =============================================================================
# 1. SETUP
# =============================================================================

# Use 'Agg' backend for non-interactive plotting
matplotlib.use('Agg') 
mne.set_log_level('WARNING')

def get_plots(raw: mne.io.Raw, step: str, 
              scalings: dict = {'eeg': 40e-6}, 
              xscale: str = 'linear', 
              channel_idx: list = [0]) -> plt.Figure:
    """
    Generates a summary figure containing Raw traces, PSD, and TFR plots.
    """
    
    # 1. Plot Raw Signal
    def plot_raw_img(raw, scalings):
        n_ch = len(raw.ch_names)
        with mne.viz.use_browser_backend('matplotlib'):
            fig = raw.plot(n_channels=n_ch, scalings=scalings, title=step, 
                           show_scrollbars=False, show=False, duration=10.0)
            
            fig.canvas.draw()
            rgba = np.asarray(fig.canvas.buffer_rgba())
            data = rgba[..., :3] 
            plt.close(fig)
        return data

    # 2. Plot Power Spectral Density (PSD)
    def plot_psd_img(raw, xscale):
        # AANGEPAST: fmax verlaagd naar 50Hz, omdat we low-pass filteren op 44Hz.
        fig = raw.compute_psd(fmin=1, fmax=50).plot(picks='eeg', show=False)
        
        fig.canvas.draw()
        rgba = np.asarray(fig.canvas.buffer_rgba())
        data = rgba[..., :3]
        plt.close(fig)
        return data

    # 3. Plot Time-Frequency Representation (TFR)
    def plot_tfr_on_ax(raw, ax, ch_idx):
        # AANGEPAST: Focus specifieker op de ranges uit de paper (tot 40Hz de gamma band)
        freqs = np.arange(4, 45, 2) 
        n_cycles = freqs / 2.0
        
        try:
            # AANGEPAST: Duration naar 1.0 om gelijk te lopen met de 1-seconde epoching van de paper
            epochs = mne.make_fixed_length_epochs(raw, duration=1.0, overlap=0, verbose=False)
            tfr = tfr_multitaper(epochs, freqs=freqs, n_cycles=n_cycles, use_fft=True, 
                                 average=True, return_itc=False)
            
            tfr.plot([ch_idx], baseline=(None, None), mode='logratio', 
                     axes=ax, show=False, colorbar=True)
        except Exception as e:
            print(f"⚠️ TFR Plot failed: {e}")

    # --- Assemble Composite Figure ---
    img_raw = plot_raw_img(raw, scalings)
    img_psd = plot_psd_img(raw, xscale)

    fig, axes = plt.subplot_mosaic(
        [['ax_raw', 'ax_raw', 'ax_tfr'],
         ['ax_psd', 'ax_psd', 'ax_psd']],
        figsize=(20, 15)
    )
    
    axes['ax_raw'].imshow(img_raw)
    axes['ax_raw'].axis('off')
    axes['ax_raw'].set_title(f"Raw Signal - {step}\n(Note: Artifacts are preserved intentionally)", fontsize=15)

    axes['ax_psd'].imshow(img_psd)
    axes['ax_psd'].axis('off')

    target_ch = channel_idx[0] if channel_idx[0] < len(raw.ch_names) else 0
    plot_tfr_on_ax(raw, axes['ax_tfr'], target_ch)
    axes['ax_tfr'].set_title(f'Time-Freq ({raw.ch_names[target_ch]})', fontsize=12)

    fig.suptitle(f"Quality Control Report: {step}", fontsize=20)
    plt.tight_layout()
    
    return fig