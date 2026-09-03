# 🛰️ Automated RF Signal Ingestion, Modulation Recognition & Demodulation Engine

> **Smart India Hackathon (SIH 2026)**  
> **Problem Statement ID:** SIH26147 (National Technical Research Organisation - NTRO)  
> **Domain:** Signals Intelligence (SIGINT), Radio Frequency Digital Signal Processing (RF DSP), Cyber Forensics & Machine Learning

---

## 📌 Executive Summary & Problem Context

In intelligence, defence, and spectrum monitoring operations (e.g., NTRO), operators intercept non-cooperative, raw radio frequency transmissions stored in `.IQ` (In-phase/Quadrature) or `.wav` format. Historically, analyzing such unknown signals requires human signal analysts to manually inspect spectral waterfalls, guess modulation schemes, hand-tune carrier/clock recovery loops, and attempt heuristic bit extraction.

**This project delivers an automated, end-to-end signal processing and cyber forensics pipeline** that:
1. Ingests raw binary `.IQ` and audio `.wav` streams across diverse hardware formats (RTL-SDR, HackRF, BladeRF, USRP).
2. Cleans hardware artifacts (DC bias, power variations) and performs pulse shaping.
3. Automatically classifies the modulation scheme using Deep Learning (1D-CNN).
4. Executes deterministic carrier recovery (Costas Loop), symbol timing recovery (Mueller & Müller), and constellation slicing to recover raw bitstreams.
5. De-interleaves and performs Forward Error Correction (FEC) decoding.
6. Carries out cyber forensics, including frame sync detection and Shannon Entropy analysis to determine plaintext vs. encrypted ciphertexts.
7. Renders real-time spectral waterfalls, constellation diagrams, and demodulation parameters via an interactive dashboard and REST API.

---

## 🏗️ System Architecture & Data Flow

```text
       RAW RECORDING (.IQ / .WAV)
                   │
                   ▼
┌───────────────────────────────────────┐
│ 1. Ingestion & Front-End Conditioning │ ◄── [src/ingestion/binary_parser.py]
│    • Byte parsing (float32, int16...) │     [src/ingestion/normalizer.py]
│    • DC Offset removal & Unit Power   │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 2. Automatic Modulation Class. (AMC)  │ ◄── [src/amc/models/cnn1d_classifier.py]
│    • 1D-CNN Feature Extractor         │
│    • Output: BPSK, QPSK, 16-QAM, FSK  │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 3. Deterministic DSP & Synchronization│ ◄── [src/dsp/synchronization/rrc_filter.py]
│    • RRC Matched Filtering            │     [src/dsp/synchronization/mueller_muller.py]
│    • Mueller & Müller Clock Recovery  │     [src/dsp/synchronization/costas_loop.py]
│    • Costas Loop Carrier Recovery     │     [src/dsp/demodulators/]
│    • Gray-Coded Decision Slicing      │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 4. De-Interleaving & Error Correction │ ◄── [src/decoding/deinterleaver.py]
│    • Matrix De-interleaving           │     [src/decoding/fec/viterbi.py]
│    • Viterbi / Reed-Solomon Decoders  │     [src/decoding/fec/reed_solomon.py]
└──────────────────┬────────────────────┘
                   │
                   ▼
┌───────────────────────────────────────┐
│ 5. Cyber Forensics & Entropy Analysis │ ◄── [src/decoding/cyber/sync_detector.py]
│    • Hamming Sync-Word Correlation    │     [src/decoding/cyber/entropy.py]
│    • Shannon Entropy Evaluation       │
└──────────────────┬────────────────────┘
                   │
                   ▼
┌──────────────────────────────────────────────────────────┐
│ 6. Interactive Dashboards & Asynchronous Service API     │
│    • Streamlit GUI (Waterfall, Constellation, Metrics)   │ ◄── [src/visualization/app_streamlit.py]
│    • FastAPI Endpoints (/health, /api/v1/signal/process) │ ◄── [src/api/app.py]
└──────────────────────────────────────────────────────────┘
```

---

## ✅ Implemented Features

### 1. Ingestion & Preprocessing Layer
- [x] **Universal Binary I/Q Stream Ingestion** ([`src/ingestion/binary_parser.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/ingestion/binary_parser.py)):
  - Parses raw interleaved I/Q binary recordings (`FLOAT32`, `FLOAT64`, `INT16` for BladeRF/HackRF, `INT8`, `UINT8` for RTL-SDR).
  - Handles frame offsets, chunking, and sample limits.
- [x] **WAV Audio Signal Parsing** ([`src/ingestion/binary_parser.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/ingestion/binary_parser.py)):
  - Stereo WAV mapping ($Ch_0 \rightarrow I, Ch_1 \rightarrow Q$).
  - Mono WAV analytic signal conversion via Hilbert Transform ($s(t) = r(t) + j \cdot \mathcal{H}\{r(t)\}$).
- [x] **Statistical Normalization & Conditioning** ([`src/ingestion/normalizer.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/ingestion/normalizer.py)):
  - Baseband DC offset removal (carrier feedthrough and ADC bias mitigation).
  - Unit power (RMS = 1.0) and peak envelope normalizers.

### 2. Deep Learning Automatic Modulation Classification (AMC)
- [x] **1D-CNN Modulation Classifier Architecture** ([`src/amc/models/cnn1d_classifier.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/amc/models/cnn1d_classifier.py)):
  - PyTorch 3-block 1D-CNN with Conv1D, BatchNorm, ReLU, MaxPool1D, and Dropout layers.
  - Takes raw 2-channel I/Q input tensors `[Batch, 2, Sequence_Length]`.
  - Supports inference mode with softmax class probability distributions for BPSK, QPSK, 16-QAM, and FSK.

### 3. Digital Signal Processing & Synchronization
- [x] **Pulse Shaping & Matched Filtering** ([`src/dsp/synchronization/rrc_filter.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/dsp/synchronization/rrc_filter.py)):
  - Closed-form discrete Root-Raised-Cosine (RRC) FIR tap generation with arbitrary roll-off ($\alpha$) and symbol spans.
  - Independent I/Q matched filtering for Inter-Symbol Interference (ISI) rejection.
- [x] **Clock & Timing Recovery** ([`src/dsp/synchronization/mueller_muller.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/dsp/synchronization/mueller_muller.py)):
  - Decision-directed Mueller & Müller discrete-time clock recovery with linear fractional-delay interpolation.
- [x] **Carrier Phase & Frequency Tracking** ([`src/dsp/synchronization/costas_loop.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/dsp/synchronization/costas_loop.py)):
  - 2nd-order discrete Phase-Locked Loop (Costas Loop) for suppressed-carrier phase derotation and frequency offset (CFO) compensation across 2-PSK, 4-PSK, and M-PSK.
- [x] **Modulation Demodulators** ([`src/dsp/demodulators/`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/dsp/demodulators/)):
  - **M-PSK Demodulator** ([`psk.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/dsp/demodulators/psk.py)): Full coherent chain (RRC $\rightarrow$ M&M $\rightarrow$ Costas $\rightarrow$ Slicing) with Gray code mapping for BPSK, QPSK, and 8PSK, plus $M_2M_4$ SNR estimation.
  - **M-QAM Demodulator** ([`qam.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/dsp/demodulators/qam.py)): Matched filtering, AGC scaling, and 16-QAM decision boundary slicing.
  - **FSK Demodulator** ([`fsk.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/dsp/demodulators/fsk.py)): Non-coherent quadrature delay-and-multiply frequency discriminator for 2-FSK.

### 4. Cyber Forensics & Decoding
- [x] **Frame Synchronization Detector** ([`src/decoding/cyber/sync_detector.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/decoding/cyber/sync_detector.py)):
  - Sliding-window bit correlator with Hamming distance error thresholding to detect packet preambles and sync words in noisy bitstreams.
- [x] **Shannon Entropy Analyzer** ([`src/decoding/cyber/entropy.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/decoding/cyber/entropy.py)):
  - Evaluates empirical Shannon entropy $H(X)$ per byte ($0.0 - 8.0\text{ bits/byte}$) to differentiate plaintext, structured telemetry, and encrypted/compressed payloads.
- [x] **Matrix De-interleaving & FEC Stubs** ([`src/decoding/deinterleaver.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/decoding/deinterleaver.py), [`src/decoding/fec/`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/decoding/fec/)):
  - Block/matrix de-interleaving to disperse burst errors.
  - Architecture skeletons for Viterbi ($K=7, \text{Rate } 1/2$) and Reed-Solomon (RS 255/223) decoders.

### 5. Frontend & REST API Interfaces
- [x] **Interactive Streamlit Web Dashboard** ([`src/visualization/app_streamlit.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/visualization/app_streamlit.py)):
  - Interactive file uploader for `.IQ` / `.wav` recordings.
  - Configurable sampling rate ($F_s$), symbol rate ($R_s$), and analysis window size.
  - Real-time Plotly FFT spectrogram waterfall heatmap and baseband I/Q constellation scatter plot.
  - Interactive execution of the demodulation chain with recovered bitstream previews.
- [x] **FastAPI Asynchronous Microservice** ([`src/api/app.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/api/app.py)):
  - `/health` health-check endpoint.
  - `/api/v1/signal/process` multipart upload endpoint returning JSON analysis (SNR, carrier frequency offset, phase error, recovered bitstream).

---

## 🚀 What Needs to Be Implemented / Roadmap

To transition this hackathon prototype into a battle-ready SIGINT platform, the following features and improvements are planned for implementation:

### 1. High Priority (Core DSP & AI Upgrades)
- [ ] **Pretrained AMC Weights & Automated Pipeline Integration**:
  - Train the `Modulation1DCNN` on RadioML2018.01A / synthetic augmented dataset (varying SNR from $-20\text{ dB}$ to $+20\text{ dB}$).
  - Export trained weights (`.pt` / ONNX) and wire the model directly into `app_streamlit.py` and `app.py` for automated zero-touch modulation selection.
- [ ] **Full Production Viterbi & Reed-Solomon FEC Decoders**:
  - Implement full Soft/Hard-decision Viterbi trellis decoding (e.g., using `numba` acceleration or `libfec` bindings) in [`viterbi.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/decoding/fec/viterbi.py).
  - Implement Galois Field $GF(2^8)$ Berlekamp-Massey / Euclidean decoding algorithm in [`reed_solomon.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/decoding/fec/reed_solomon.py) to correct up to $t = (n-k)/2$ corrupted bytes.
- [ ] **Advanced Modulation Demodulators**:
  - 64-QAM and 256-QAM soft/hard-decision slicers in [`qam.py`](file:///c:/Users/harsh/OneDrive/Desktop/K%20S%20Harshitaa/Projects/Hackathons/SIH%202026/src/dsp/demodulators/qam.py).
  - 4-FSK / M-FSK multi-tone filters, Continuous Phase Frequency Shift Keying (CPFSK), and GMSK (Gaussian Minimum Shift Keying) for tactical radio interception.
  - OFDM (Orthogonal Frequency Division Multiplexing) channel estimation and subcarrier demapping.

### 2. Medium Priority (Forensics & Automated Intelligence)
- [ ] **Protocol Header Parsers & Bitstream Telemetry Decoders**:
  - Standard frame deframers for common protocols: CCSDS (Space telemetry), AX.25 (Packet radio), ADS-B (1090 MHz aircraft tracking), AIS (Marine VHF).
  - Hex dump and ASCII text decoding views in the Streamlit UI.
- [ ] **Cryptographic Heuristic Detection**:
  - Advanced randomness tests (NIST SP 800-22 test suite subset, Chi-Square goodness-of-fit, Serial Autocorrelation) to detect weak ciphers, LFSR-scrambled data, or AES encryption.
- [ ] **Blind Symbol Rate & Carrier Frequency Estimator**:
  - Implement Cyclostationary feature analysis / Spectral Correlation Function (SCF) or Wavelet transforms to automatically extract symbol rate ($R_s$) without user manual input.

### 3. Pipeline, Testing & DevOps
- [ ] **Automated End-to-End Synthetic Benchmark Suite (`tests/`)**:
  - Synthetic signal generator module creating AWGN/Rayleigh fading channels with known CFO and timing offsets.
  - BER vs. SNR waterfall curves validating Viterbi and RS performance against theoretical limits.
- [ ] **Hardware SDR Live Streaming Integration**:
  - Real-time streaming ingestion via `pyrtlsdr`, `SoapySDR`, or UHD for live HackRF/RTL-SDR antenna capture.
- [ ] **Docker Containerization**:
  - Multi-stage `Dockerfile` and `docker-compose.yml` bundling FastAPI backend, Streamlit frontend, and GPU PyTorch acceleration.

---

## 👥 Team Roles & Responsibilities

| Team Member | Domain | Module Ownership | Core Deliverables |
| :--- | :--- | :--- | :--- |
| **Member 1** | VLSI / EE | `src/ingestion/`, `src/dsp/synchronization/` | Binary .IQ parsing, DC offset removal, RRC matched filters, Costas Loop. |
| **Member 2** | CSE Core | `src/amc/`, `src/visualization/`, `src/api/` | 1D-CNN modulation classification model, Streamlit GUI, FastAPI backend. |
| **Member 3** | Cyber 1 | `src/dsp/demodulators/`, `src/dsp/synchronization/` | Mueller & Müller timing recovery, PSK/QAM/FSK slicing, SNR estimators. |
| **Member 4** | Cyber 2 | `src/decoding/fec/`, `src/decoding/` | Matrix de-interleaver, Viterbi decoder, Reed-Solomon algebraic solver. |
| **Member 5** | Cyber 3 | `src/decoding/cyber/` | Hamming sync word correlation, Shannon entropy analysis, packet deframing. |
| **Member 6** | Lead / Arch | `tests/`, System Integration | End-to-end integration, BER benchmarking, presentation, and deployment. |

---

## 💻 Installation & Quickstart

### Prerequisites
- Python 3.10+ (Recommended: Python 3.11)
- Virtual environment tool (`venv` or `conda`)

### 1. Clone & Setup Environment
```bash
# Clone the repository
git clone <repo-url>
cd "SIH 2026"

# Create and activate virtual environment
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Launch Streamlit Web Application
```bash
streamlit run src/visualization/app_streamlit.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser to interact with the GUI.

### 3. Launch FastAPI Backend
```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000 --reload
```
API Documentation and interactive Swagger UI will be live at: [http://localhost:8000/docs](http://localhost:8000/docs).

---

## 📁 Repository Structure

```text
SIH 2026/
├── generate_pdf.py                      # PDF generation script for team guides
├── requirements.txt                     # Project dependencies
├── README.md                            # Comprehensive project documentation
├── SIH2026_Plain_English_Team_Guide.pdf # Plain-English team reference
├── SIH2026_Project_Execution_Guide.pdf  # Technical architecture execution guide
└── src/
    ├── amc/                             # Automatic Modulation Classification (AI)
    │   └── models/
    │       └── cnn1d_classifier.py      # 1D-CNN PyTorch model
    ├── api/                             # FastAPI REST API Backend
    │   └── app.py                       # Endpoints for signal processing
    ├── decoding/                        # De-interleaving, FEC & Forensics
    │   ├── cyber/
    │   │   ├── entropy.py               # Shannon entropy metric calculator
    │   │   └── sync_detector.py         # Hamming distance sync-word correlator
    │   ├── fec/
    │   │   ├── reed_solomon.py          # RS block decoder
    │   │   └── viterbi.py               # Viterbi convolutional decoder
    │   └── deinterleaver.py             # Matrix/block de-interleaver
    ├── dsp/                             # Deterministic DSP & Synchronization
    │   ├── base.py                      # Abstract base classes and data models
    │   ├── demodulators/
    │   │   ├── fsk.py                   # 2-FSK non-coherent discriminator
    │   │   ├── psk.py                   # BPSK/QPSK/8PSK coherent demodulator
    │   │   └── qam.py                   # 16-QAM / 64-QAM demodulator
    │   └── synchronization/
    │       ├── costas_loop.py           # Carrier phase & frequency recovery
    │       ├── mueller_muller.py        # Symbol clock timing recovery
    │       └── rrc_filter.py            # Root-Raised-Cosine FIR filter
    ├── ingestion/                       # Signal Ingestion & Conditioning
    │   ├── binary_parser.py             # Raw .IQ / .wav file parser
    │   └── normalizer.py                # DC offset removal and RMS normalizer
    └── visualization/                   # Dashboards
        └── app_streamlit.py             # Interactive Streamlit GUI
```

---

## 📜 License & Acknowledgments
Developed for **Smart India Hackathon 2026** under Problem Statement **SIH26147**.  
Built with PyTorch, SciPy, NumPy, Streamlit, and FastAPI.
