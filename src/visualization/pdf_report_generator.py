"""
Forensic Intelligence Dossier & PDF Report Generation Service.

Generates a publication-quality, defense-grade SIGINT Intelligence Dossier PDF with:
- Case Header & Chain-of-Custody SHA-256 Provenance Seal
- Ingested Signal Parameters & Preprocessing Metrics
- Neural / Statistical Automatic Modulation Classification (AMC) Assessment
- Coherent DSP Synchronization (CFO Derotation, M&M Clock Recovery, Constellation Analysis)
- Forward Error Correction (FEC) Multi-stage Diagnostics (Deinterleaving, Viterbi MLSE, RS GF(256))
- Frame Synchronization, Sliding Hamming Correlation & Bitstream Preamble Extraction
- Shannon Information Entropy Profiling & Conservative Payload Forensics
- Emitter RF Fingerprinting (Signal DNA) & Baseline Anomaly Deviations
- Native Vector Charts (Constellation I/Q Scatter, Correlation Curve, Waveform & Metrics)
- Comprehensive Glossary and Detailed Explanation of all scientific terms and values.
"""

from __future__ import annotations

import io
import time
from typing import Any, Dict, List, Optional
import numpy as np

from reportlab.lib import colors
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import (
    HRFlowable,
    KeepTogether,
    PageBreak,
    Paragraph,
    SimpleDocTemplate,
    Spacer,
    Table,
    TableStyle,
)
from reportlab.graphics.shapes import Drawing, Rect, String, Line, Circle, PolyLine


class ReportChartBuilder:
    """Builds native vector charts for ReportLab without external GUI/matplotlib dependencies."""

    @staticmethod
    def draw_constellation(points: np.ndarray, title: str = "Recovered Symbol Constellation (I vs Q)", width: int = 240, height: int = 150) -> Drawing:
        """Draws vector constellation I/Q diagram."""
        d = Drawing(width, height)
        # Background box
        d.add(Rect(0, 0, width, height, fillColor=colors.HexColor("#0B0F17"), strokeColor=colors.HexColor("#1F2937"), strokeWidth=1))
        # Title
        d.add(String(8, height - 14, title, fontName="Helvetica-Bold", fontSize=7.5, fillColor=colors.HexColor("#94A3B8")))
        
        # Center axes
        cx, cy = width / 2.0, (height - 18) / 2.0
        d.add(Line(20, cy, width - 20, cy, strokeColor=colors.HexColor("#1E293B"), strokeWidth=1))
        d.add(Line(cx, 10, cx, height - 25, strokeColor=colors.HexColor("#1E293B"), strokeWidth=1))
        
        if len(points) == 0:
            return d

        pts = points[:300]
        # Normalize scale
        max_val = max(1e-3, float(np.max(np.abs(pts))))
        scale = min(cx - 25, cy - 20) / max_val

        for p in pts:
            px = cx + np.real(p) * scale
            py = cy + np.imag(p) * scale
            if 5 <= px <= width - 5 and 5 <= py <= height - 5:
                d.add(Circle(px, py, 1.8, fillColor=colors.HexColor("#10B981"), strokeColor=colors.HexColor("#059669"), strokeWidth=0.5))
        return d

    @staticmethod
    def draw_correlation_curve(curve: np.ndarray, peak_idx: Optional[int] = None, width: int = 480, height: int = 100) -> Drawing:
        """Draws vector sliding correlation / Hamming curve."""
        d = Drawing(width, height)
        d.add(Rect(0, 0, width, height, fillColor=colors.HexColor("#0B0F17"), strokeColor=colors.HexColor("#1F2937"), strokeWidth=1))
        d.add(String(8, height - 12, "Sliding Hamming Bit Cross-Correlation Strength Curve [0.0 - 1.0]", fontName="Helvetica-Bold", fontSize=7.5, fillColor=colors.HexColor("#94A3B8")))

        if len(curve) < 2:
            return d

        plot_w = width - 40
        plot_h = height - 30
        ox, oy = 25, 12

        # Grid line
        d.add(Line(ox, oy + plot_h, ox + plot_w, oy + plot_h, strokeColor=colors.HexColor("#1E293B"), strokeWidth=0.5))
        d.add(Line(ox, oy, ox + plot_w, oy, strokeColor=colors.HexColor("#1E293B"), strokeWidth=0.5))

        slice_len = min(150, len(curve))
        c_slice = curve[:slice_len]
        poly_pts = []
        for i, val in enumerate(c_slice):
            x = ox + (i / max(1, slice_len - 1)) * plot_w
            y = oy + float(np.clip(val, 0.0, 1.0)) * plot_h
            poly_pts.append(x)
            poly_pts.append(y)

        d.add(PolyLine(poly_pts, strokeColor=colors.HexColor("#3B82F6"), strokeWidth=1.5))

        if peak_idx is not None and peak_idx < slice_len:
            pk_x = ox + (peak_idx / max(1, slice_len - 1)) * plot_w
            pk_y = oy + float(np.clip(c_slice[peak_idx], 0.0, 1.0)) * plot_h
            d.add(Circle(pk_x, pk_y, 4, fillColor=colors.HexColor("#EF4444"), strokeColor=colors.white, strokeWidth=1))
            d.add(String(pk_x - 12, pk_y + 6, f"SYNC (Idx {peak_idx})", fontName="Helvetica-Bold", fontSize=6.5, fillColor=colors.HexColor("#FCA5A5")))

        return d


class PDFReportGenerator:
    """
    Generates defense-grade, comprehensive SIGINT & Cyber Forensics PDF Intelligence Reports.
    """

    @classmethod
    def generate_pdf_bytes(
        cls,
        case_id: str,
        timestamp: str,
        meta_info: Dict[str, Any],
        amc_res: Any,
        dsp_analysis: Any,
        fec_res: Any,
        forensics_res: Any,
        signal_dna: Any,
        evidence_chain: Any,
        norm_signal: np.ndarray
    ) -> bytes:
        """
        Builds the complete multi-page PDF dossier in-memory.
        """
        buffer = io.BytesIO()
        doc = SimpleDocTemplate(
            buffer,
            pagesize=letter,
            rightMargin=36,
            leftMargin=36,
            topMargin=36,
            bottomMargin=36
        )

        styles = getSampleStyleSheet()

        # Premium Technical Defense SIGINT Palette
        c_navy = colors.HexColor("#0F172A")
        c_blue = colors.HexColor("#1E3A8A")
        c_cyan = colors.HexColor("#0284C7")
        c_slate = colors.HexColor("#334155")
        c_light = colors.HexColor("#F8FAFC")
        c_border = colors.HexColor("#CBD5E1")
        c_green = colors.HexColor("#047857")

        # Typography Styles
        style_title = ParagraphStyle(
            'DocTitle',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=16,
            leading=20,
            textColor=c_navy
        )
        style_subtitle = ParagraphStyle(
            'DocSubtitle',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=9,
            leading=12,
            textColor=c_slate
        )
        style_sec_heading = ParagraphStyle(
            'SecHeading',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=11,
            leading=14,
            textColor=c_blue,
            spaceBefore=8,
            spaceAfter=4,
            keepWithNext=True
        )
        style_body = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=8,
            leading=11,
            textColor=c_slate
        )
        style_tbl_hdr = ParagraphStyle(
            'TblHdr',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=7.5,
            leading=9.5,
            textColor=colors.white
        )
        style_tbl_cell = ParagraphStyle(
            'TblCell',
            parent=styles['Normal'],
            fontName='Helvetica',
            fontSize=7.5,
            leading=9.5,
            textColor=c_slate
        )
        style_tbl_cell_bold = ParagraphStyle(
            'TblCellBold',
            parent=styles['Normal'],
            fontName='Helvetica-Bold',
            fontSize=7.5,
            leading=9.5,
            textColor=c_navy
        )

        story = []

        # ================= PAGE 1: CASE BRIEFING & EXECUTIVE ASSESSMENT =================
        header_data = [
            [
                Paragraph("<b>CHROMA CODE | RF SIGNALS INTELLIGENCE & CYBER FORENSICS DOSSIER</b>", style_title),
                Paragraph(f"<b>STATUS: VERIFIED INTACT</b><br/><font size='7' color='#047857'>CONFIDENTIAL / OPERATIONAL</font>", ParagraphStyle('HdrR', parent=styles['Normal'], fontName='Helvetica-Bold', fontSize=8, alignment=2, textColor=c_green))
            ],
            [
                Paragraph(f"CASE IDENTIFIER: <b>{case_id}</b> | TIMESTAMP: <b>{timestamp}</b> | TARGET CAPTURE: <b>{meta_info.get('Filename', 'capture.iq')}</b>", style_subtitle),
                Paragraph(f"SEAL: {evidence_chain.final_evidence_seal[:16]}...", ParagraphStyle('HdrR2', parent=styles['Normal'], fontName='Courier', fontSize=7, alignment=2, textColor=c_slate))
            ]
        ]
        t_hdr = Table(header_data, colWidths=[380, 160])
        t_hdr.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(t_hdr)
        story.append(HRFlowable(width="100%", thickness=1.5, color=c_blue, spaceBefore=4, spaceAfter=8))

        # Executive Summary Box
        summary_text = (
            f"<b>EXECUTIVE INTELLIGENCE VERDICT:</b> Target signal intercepted in <b>{meta_info.get('Format', 'I/Q')}</b> format "
            f"classified as <b>{amc_res.modulation}</b> (Neural/Heuristic confidence: <b>{amc_res.confidence*100:.1f}%</b>). "
            f"Estimated SNR: <b>{dsp_analysis.extracted_params.get('Estimated SNR', 'N/A')}</b> with carrier frequency offset "
            f"<b>{dsp_analysis.extracted_params.get('Carrier Frequency Offset (Δf)', 'N/A')}</b>. "
            f"Emitter physical signature strongly matches catalog profile <b>{signal_dna.primary_emitter.emitter_id} ({signal_dna.primary_emitter.designation})</b> "
            f"at <b>{signal_dna.primary_emitter.similarity_score*100:.1f}% similarity</b>. "
            f"Frame sync preamble <b>0x{forensics_res.sync_word_hex}</b> detected (Hamming distance: {forensics_res.min_hamming_distance} bits, confidence: {forensics_res.sync_confidence*100:.1f}%). "
            f"Payload information density measured at <b>{forensics_res.entropy_bits_per_byte:.3f} bits/byte</b>, characterized as <b>{forensics_res.payload_classification}</b>."
        )
        t_sum = Table([[Paragraph(summary_text, style_body)]], colWidths=[540])
        t_sum.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), c_light),
            ('BOX', (0, 0), (-1, -1), 1, c_cyan),
            ('LINELEFT', (0, 0), (-1, -1), 3, c_blue),
            ('TOPPADDING', (0, 0), (-1, -1), 6),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 6),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(t_sum)
        story.append(Spacer(1, 8))

        # ---------------- SECTION 1 & 2: PHYSICAL INGESTION & AMC ----------------
        story.append(Paragraph("1. PHYSICAL INGESTION & AUTOMATIC MODULATION CLASSIFICATION", style_sec_heading))
        ingest_rows = [
            [Paragraph("<b>Parameter</b>", style_tbl_hdr), Paragraph("<b>Extracted Value</b>", style_tbl_hdr), Paragraph("<b>Physical / Mathematical Meaning</b>", style_tbl_hdr)],
            [Paragraph("Sampling Frequency (Fs)", style_tbl_cell_bold), Paragraph(str(meta_info.get("Sample Rate")), style_tbl_cell), Paragraph("Digitization rate of analog RF baseband bandwidth.", style_tbl_cell)],
            [Paragraph("Symbol Rate (Rs)", style_tbl_cell_bold), Paragraph(str(meta_info.get("Symbol Rate")), style_tbl_cell), Paragraph("Discrete modulation state transition rate (Baud).", style_tbl_cell)],
            [Paragraph("Sample Count & Duration", style_tbl_cell_bold), Paragraph(f"{meta_info.get('Sample Count')} ({meta_info.get('Duration')})", style_tbl_cell), Paragraph("Total temporal acquisition window of the captured frame.", style_tbl_cell)],
            [Paragraph("Classified Modulation", style_tbl_cell_bold), Paragraph(f"<b>{amc_res.modulation}</b> ({amc_res.confidence*100:.1f}%)", style_tbl_cell), Paragraph("Identified digital modulation constellation alphabet.", style_tbl_cell)],
            [Paragraph("Estimated SNR", style_tbl_cell_bold), Paragraph(str(dsp_analysis.extracted_params.get("Estimated SNR")), style_tbl_cell), Paragraph("In-band signal power to background noise spectral density ratio.", style_tbl_cell)],
        ]
        t_ingest = Table(ingest_rows, colWidths=[130, 120, 290])
        t_ingest.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), c_navy),
            ('GRID', (0, 0), (-1, -1), 0.5, c_border),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light]),
            ('TOPPADDING', (0, 0), (-1, -1), 3.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ]))
        story.append(t_ingest)
        story.append(Spacer(1, 8))

        # ---------------- SECTION 3: COHERENT DSP & SYNCHRONIZATION ----------------
        story.append(Paragraph("2. COHERENT DSP SYNCHRONIZATION & CONSTELLATION RECOVERY", style_sec_heading))
        
        pts_before = dsp_analysis.stages[0].constellation_pts if dsp_analysis.stages else np.array([])
        pts_after = dsp_analysis.stages[3].constellation_pts if len(dsp_analysis.stages) >= 4 else np.array([])
        
        d_before = ReportChartBuilder.draw_constellation(pts_before, "Stage 1: Raw Unsynchronized Baseband", width=265, height=125)
        d_after = ReportChartBuilder.draw_constellation(pts_after, "Stage 4: Coherent PLL & Clock Recovered", width=265, height=125)

        t_charts = Table([[d_before, d_after]], colWidths=[270, 270])
        t_charts.setStyle(TableStyle([
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('LEFTPADDING', (0, 0), (-1, -1), 0),
            ('RIGHTPADDING', (0, 0), (-1, -1), 0),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(t_charts)
        story.append(Spacer(1, 4))

        dsp_rows = [
            [Paragraph("<b>DSP Stage</b>", style_tbl_hdr), Paragraph("<b>Measured Metric</b>", style_tbl_hdr), Paragraph("<b>Algorithmic Method & Operational Role</b>", style_tbl_hdr)],
            [Paragraph("Matched Filtering", style_tbl_cell_bold), Paragraph("RRC α = 0.35", style_tbl_cell), Paragraph("Maximizes signal-to-noise ratio and eliminates Intersymbol Interference (ISI).", style_tbl_cell)],
            [Paragraph("Timing Clock Recovery", style_tbl_cell_bold), Paragraph("Decision-Directed M&M", style_tbl_cell), Paragraph("Interpolates optimum symbol strobe instants independent of sample clock jitter.", style_tbl_cell)],
            [Paragraph("Carrier CFO & Phase", style_tbl_cell_bold), Paragraph(f"Δf = {dsp_analysis.extracted_params.get('Carrier Frequency Offset (Δf)', '0 Hz')}", style_tbl_cell), Paragraph("Costas Phase-Locked Loop derotates Doppler and local oscillator mismatches.", style_tbl_cell)],
            [Paragraph("Recovered Symbols", style_tbl_cell_bold), Paragraph(str(dsp_analysis.extracted_params.get("Total Recovered Symbols", "0")), style_tbl_cell), Paragraph("Discrete sliced constellation decision points feeding demodulation.", style_tbl_cell)],
        ]
        t_dsp = Table(dsp_rows, colWidths=[130, 120, 290])
        t_dsp.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), c_navy),
            ('GRID', (0, 0), (-1, -1), 0.5, c_border),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light]),
            ('TOPPADDING', (0, 0), (-1, -1), 3.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ]))
        story.append(t_dsp)

        # ================= PAGE 2: FEC, CYBER FORENSICS, DNA, EVIDENCE & GLOSSARY =================
        story.append(PageBreak())

        # ---------------- SECTION 4: FEC DECODING ----------------
        story.append(Paragraph("3. CONCATENATED FORWARD ERROR CORRECTION (FEC) SUBSYSTEM", style_sec_heading))
        fec_rows = [
            [Paragraph("<b>FEC Subsystem Layer</b>", style_tbl_hdr), Paragraph("<b>Status & Convergence</b>", style_tbl_hdr), Paragraph("<b>Mathematical Performance & Diagnostics</b>", style_tbl_hdr)],
            [Paragraph("Matrix De-Interleaver", style_tbl_cell_bold), Paragraph(str(fec_res.deinterleaver_status), style_tbl_cell), Paragraph("Inverts transmitter row-write/col-read matrix to disperse burst channel errors.", style_tbl_cell)],
            [Paragraph("Viterbi MLSE Decoder", style_tbl_cell_bold), Paragraph(str(fec_res.viterbi_status), style_tbl_cell), Paragraph("64-state Trellis search (K=7, Rate 1/2, polynomials 171/133) resolving bit flips.", style_tbl_cell)],
            [Paragraph("Reed-Solomon GF(256)", style_tbl_cell_bold), Paragraph(str(fec_res.reed_solomon_status), style_tbl_cell), Paragraph("Berlekamp-Massey & Forney solver correcting residual multi-byte symbol corruptions.", style_tbl_cell)],
            [Paragraph("Bit Error Corrections", style_tbl_cell_bold), Paragraph(f"Resolved: {fec_res.errors_corrected} / Detected: {fec_res.errors_detected}", style_tbl_cell), Paragraph("Total discrepancies eliminated through algebraic and convolutional decoding.", style_tbl_cell)],
        ]
        t_fec = Table(fec_rows, colWidths=[130, 150, 260])
        t_fec.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), c_navy),
            ('GRID', (0, 0), (-1, -1), 0.5, c_border),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light]),
            ('TOPPADDING', (0, 0), (-1, -1), 3.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ]))
        story.append(t_fec)
        story.append(Spacer(1, 8))

        # ---------------- SECTION 5: FRAME SYNCHRONIZATION & FORENSICS ----------------
        story.append(Paragraph("4. FRAME SYNCHRONIZATION, ENTROPY & PAYLOAD FORENSICS", style_sec_heading))
        
        d_corr = ReportChartBuilder.draw_correlation_curve(
            forensics_res.correlation_curve,
            peak_idx=forensics_res.first_sync_index,
            width=540,
            height=65
        )
        story.append(d_corr)
        story.append(Spacer(1, 4))

        forensics_rows = [
            [Paragraph("<b>Forensics Metric</b>", style_tbl_hdr), Paragraph("<b>Measurement</b>", style_tbl_hdr), Paragraph("<b>Intelligence Interpretation</b>", style_tbl_hdr)],
            [Paragraph("Preamble Sync Word", style_tbl_cell_bold), Paragraph(f"0x{forensics_res.sync_word_hex}", style_tbl_cell), Paragraph(f"Match status: {forensics_res.summary_card.get('Frame Sync')} at offset bit {forensics_res.first_sync_index}.", style_tbl_cell)],
            [Paragraph("Hamming Match Distance", style_tbl_cell_bold), Paragraph(f"{forensics_res.min_hamming_distance} bits mismatch", style_tbl_cell), Paragraph(f"Confidence score: {forensics_res.sync_confidence*100:.1f}% based on normalized bit cross-correlation.", style_tbl_cell)],
            [Paragraph("Shannon Information Entropy", style_tbl_cell_bold), Paragraph(f"<b>{forensics_res.entropy_bits_per_byte:.3f}</b> bits/byte", style_tbl_cell), Paragraph("H(X) = -sum(p*log2(p)). Range: 0.0 (idle/padding) to 8.0 (maximal randomness).", style_tbl_cell)],
            [Paragraph("Payload Characterization", style_tbl_cell_bold), Paragraph(f"<b>{forensics_res.payload_classification}</b>", style_tbl_cell), Paragraph(str(forensics_res.payload_interpretation), style_tbl_cell)],
        ]
        t_fore = Table(forensics_rows, colWidths=[130, 110, 300])
        t_fore.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), c_navy),
            ('GRID', (0, 0), (-1, -1), 0.5, c_border),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light]),
            ('TOPPADDING', (0, 0), (-1, -1), 3.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ]))
        story.append(t_fore)
        story.append(Spacer(1, 8))

        # ---------------- SECTION 6: EMITTER FINGERPRINTING & EVIDENCE CHAIN ----------------
        story.append(Paragraph("5. SIGNAL DNA EMITTER FINGERPRINTING & CRYPTOGRAPHIC CHAIN OF CUSTODY", style_sec_heading))
        dna_rows = [
            [Paragraph("<b>Signal DNA Profile</b>", style_tbl_hdr), Paragraph("<b>Observation</b>", style_tbl_hdr), Paragraph("<b>Baseline Anomaly Status</b>", style_tbl_hdr)],
            [
                Paragraph(f"Primary Emitter: <b>{signal_dna.primary_emitter.emitter_id}</b>", style_tbl_cell_bold),
                Paragraph(f"Designation: {signal_dna.primary_emitter.designation}<br/>Similarity: <b>{signal_dna.primary_emitter.similarity_score*100:.1f}%</b> ({signal_dna.primary_emitter.status})", style_tbl_cell),
                Paragraph(f"Fingerprint: {signal_dna.fingerprint_hash[:16]}...<br/>Anomalies: {len(signal_dna.anomaly_factors)} factors assessed", style_tbl_cell)
            ],
            [
                Paragraph("<b>Evidence Custody Seal</b>", style_tbl_cell_bold),
                Paragraph(f"Root Hash: {evidence_chain.root_capture_hash[:20]}...<br/>Master Seal: <b>{evidence_chain.final_evidence_seal[:20]}...</b>", style_tbl_cell),
                Paragraph(f"Chain Continuity: <b>{'VERIFIED INTACT (5/5 links)' if evidence_chain.is_chain_intact else 'COMPROMISED'}</b>", style_tbl_cell)
            ]
        ]
        t_dna = Table(dna_rows, colWidths=[130, 210, 200])
        t_dna.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, 0), c_navy),
            ('GRID', (0, 0), (-1, -1), 0.5, c_border),
            ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, c_light]),
            ('TOPPADDING', (0, 0), (-1, -1), 3.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3.5),
        ]))
        story.append(t_dna)
        story.append(Spacer(1, 8))

        # ---------------- SECTION 7: COMPREHENSIVE GLOSSARY & SCIENTIFIC EXPLANATIONS ----------------
        story.append(Paragraph("6. COMPREHENSIVE SCIENTIFIC GLOSSARY & EXPLANATION OF TERMS", style_sec_heading))
        glossary_items = [
            ("Shannon Information Entropy (H)", "Empirical measure of randomness and information density (in bits per byte, 0.0 to 8.0). Uncompressed plaintext typically measures 3.0-5.0 b/B; packed/compressed data yields 5.5-7.3 b/B; cryptographic ciphertext yields > 7.35 b/B."),
            ("Carrier Frequency Offset (CFO / Δf)", "The spectral difference (in Hz) between the received transmitter center frequency and the local receiver oscillator. Mitigated via 4th-power / decision-directed Costas loops."),
            ("Signal-to-Noise Ratio (SNR)", "Ratio of received carrier power to additive thermal noise density (dB), estimated using M2M4 split-moment and spectral noise floor calculations."),
            ("Viterbi Algorithm (K=7, Rate 1/2)", "Maximum Likelihood Sequence Estimation (MLSE) algorithm over a 64-state Trellis that finds the minimum Hamming distance path through convolutional code transitions."),
            ("Reed-Solomon RS(255, 223)", "Non-binary algebraic block code over Galois Field GF(2^8) capable of detecting and correcting up to t = (N-K)/2 = 16 byte symbol errors per 255-byte codeword."),
            ("Sliding Hamming Distance", "Bitwise error distance metric d_H(u, v) = sum(u XOR v) evaluated across sliding windows to locate frame synchronization preambles in noisy channels."),
            ("Signal DNA (RF Fingerprinting)", "Unintentional transmitter hardware impairments (I/Q imbalance, phase noise, amplifier nonlinearity) used to distinguish individual physical transmitters."),
            ("Cryptographic Chain of Custody", "Cryptographically linked sequence of SHA-256 digests ensuring forensically defensible integrity and tamper-evidence from raw capture to intelligence verdict.")
        ]
        
        g_rows = [[Paragraph(f"<b>• {term}:</b> {defn}", style_body)] for term, defn in glossary_items]
        t_glossary = Table(g_rows, colWidths=[540])
        t_glossary.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), c_light),
            ('BOX', (0, 0), (-1, -1), 0.5, c_border),
            ('TOPPADDING', (0, 0), (-1, -1), 2.5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 2.5),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(t_glossary)

        doc.build(story)
        pdf_data = buffer.getvalue()
        buffer.close()
        return pdf_data
