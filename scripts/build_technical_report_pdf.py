import os, sys
from pathlib import Path
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import cm, mm
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, Image as RLImage, KeepTogether, PageBreak
)
from reportlab.pdfgen import canvas

ROOT = Path(__file__).resolve().parents[1]

class NumberedCanvas(canvas.Canvas):
    """Canvas that performs two passes to dynamically compute total page count."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super(NumberedCanvas, self).showPage()
        super(NumberedCanvas, self).save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 8)
        self.setFillColor(colors.HexColor("#60758A"))
        
        # Header (pages 2+)
        if self._pageNumber > 1:
            self.drawString(
                20 * mm,
                297 * mm - 12 * mm,
                "SIH26057 Technical Report: AI-Powered Underwater Sonar Analysis"
            )
            self.drawRightString(
                210 * mm - 20 * mm,
                297 * mm - 12 * mm,
                "MoES / NIOT"
            )
            self.setStrokeColor(colors.HexColor("#DCE7EF"))
            self.setLineWidth(0.5)
            self.line(20 * mm, 297 * mm - 14 * mm, 210 * mm - 20 * mm, 297 * mm - 14 * mm)

        # Footer (all pages)
        self.setStrokeColor(colors.HexColor("#DCE7EF"))
        self.setLineWidth(0.5)
        self.line(20 * mm, 16 * mm, 210 * mm - 20 * mm, 16 * mm)

        self.drawString(
            20 * mm,
            11 * mm,
            "Ministry of Earth Sciences (MoES) | National Institute of Ocean Technology (NIOT)"
        )
        self.drawRightString(
            210 * mm - 20 * mm,
            11 * mm,
            f"Page {self._pageNumber} of {page_count}"
        )
        self.restoreState()


def generate_pdf(output_path: str):
    doc = SimpleDocTemplate(
        output_path,
        pagesize=A4,
        leftMargin=20 * mm,
        rightMargin=20 * mm,
        topMargin=18 * mm,
        bottomMargin=20 * mm
    )

    styles = getSampleStyleSheet()

    # Academic Typography Palette
    NAVY = colors.HexColor("#16324F")
    BODY_TEXT = colors.HexColor("#222222")
    MUTED = colors.HexColor("#555555")
    LINE_GRAY = colors.HexColor("#DCE7EF")
    CARD_BG = colors.HexColor("#F8FAFC")

    # Custom Paragraph Styles
    style_inst_left = ParagraphStyle(
        'InstLeft',
        fontName='Times-Bold',
        fontSize=8.5,
        leading=11,
        textColor=NAVY,
        alignment=0
    )
    style_inst_right = ParagraphStyle(
        'InstRight',
        fontName='Times-Italic',
        fontSize=8.5,
        leading=11,
        textColor=MUTED,
        alignment=2
    )

    style_title = ParagraphStyle(
        'DocTitle',
        fontName='Times-Bold',
        fontSize=20,
        leading=24,
        textColor=NAVY,
        alignment=1,
        spaceAfter=4
    )
    style_subtitle = ParagraphStyle(
        'DocSubTitle',
        fontName='Times-Roman',
        fontSize=12,
        leading=16,
        textColor=MUTED,
        alignment=1,
        spaceAfter=3
    )
    style_problem_id = ParagraphStyle(
        'ProblemID',
        fontName='Times-Bold',
        fontSize=10.5,
        leading=14,
        textColor=NAVY,
        alignment=1,
        spaceAfter=3
    )
    style_doc_heading = ParagraphStyle(
        'DocHeading',
        fontName='Times-Bold',
        fontSize=13,
        leading=16,
        textColor=NAVY,
        alignment=1,
        spaceAfter=12
    )

    style_h1 = ParagraphStyle(
        'SectionH1',
        fontName='Times-Bold',
        fontSize=12,
        leading=15,
        textColor=NAVY,
        spaceBefore=12,
        spaceAfter=5,
        keepWithNext=True
    )
    style_h2 = ParagraphStyle(
        'SectionH2',
        fontName='Times-Bold',
        fontSize=10.5,
        leading=13.5,
        textColor=NAVY,
        spaceBefore=8,
        spaceAfter=3,
        keepWithNext=True
    )

    style_body = ParagraphStyle(
        'BodyAcademic',
        fontName='Times-Roman',
        fontSize=9.5,
        leading=13.5,
        textColor=BODY_TEXT,
        spaceAfter=6,
        alignment=4 # Justified
    )
    style_abstract_body = ParagraphStyle(
        'AbstractBody',
        fontName='Times-Italic',
        fontSize=9.0,
        leading=13.0,
        textColor=BODY_TEXT,
        alignment=4,
        leftIndent=15,
        rightIndent=15,
        spaceAfter=6
    )

    style_table_header = ParagraphStyle(
        'TH',
        fontName='Times-Bold',
        fontSize=8.5,
        leading=11,
        textColor=NAVY,
        alignment=0
    )
    style_table_cell = ParagraphStyle(
        'TD',
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        textColor=BODY_TEXT,
        alignment=0
    )
    style_table_cell_bold = ParagraphStyle(
        'TDBold',
        fontName='Times-Bold',
        fontSize=8.5,
        leading=11,
        textColor=BODY_TEXT,
        alignment=0
    )
    style_table_cell_right = ParagraphStyle(
        'TDRight',
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        textColor=BODY_TEXT,
        alignment=2
    )

    style_caption = ParagraphStyle(
        'FigCaption',
        fontName='Times-Italic',
        fontSize=8.5,
        leading=11,
        textColor=MUTED,
        alignment=1,
        spaceBefore=4,
        spaceAfter=10
    )

    story = []

    # ---------------------------------------------------------
    # PAGE 1: INSTITUTIONAL HEADER & TITLE BLOCK
    # ---------------------------------------------------------
    header_data = [
        [
            Paragraph("<b>Ministry of Earth Sciences</b><br/>Government of India", style_inst_left),
            Paragraph("<b>National Institute of Ocean Technology</b><br/>Technology for a Safer Ocean", style_inst_right)
        ]
    ]
    t_header = Table(header_data, colWidths=[85 * mm, 85 * mm])
    t_header.setStyle(TableStyle([
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
        ('TOPPADDING', (0, 0), (-1, -1), 0),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(t_header)
    story.append(Spacer(1, 2 * mm))
    story.append(HRFlowable(width="100%", thickness=1.0, color=NAVY, spaceAfter=8))

    story.append(Paragraph("MARINE ANOMALY INTELLIGENCE", style_title))
    story.append(Paragraph("AI-Powered Underwater Sonar Analysis", style_subtitle))
    story.append(Paragraph("Problem Statement: SIH26057", style_problem_id))
    story.append(Paragraph("TECHNICAL REPORT", style_doc_heading))

    # Metadata Table
    meta_table_data = [
        [Paragraph("Organization", style_table_cell_bold), Paragraph("Ministry of Earth Sciences (MoES)", style_table_cell)],
        [Paragraph("Department", style_table_cell_bold), Paragraph("National Institute of Ocean Technology (NIOT)", style_table_cell)],
        [Paragraph("Problem ID", style_table_cell_bold), Paragraph("SIH26057", style_table_cell)],
        [Paragraph("System", style_table_cell_bold), Paragraph("AI-Powered Underwater Marine Debris and Anomaly Detection", style_table_cell)],
        [Paragraph("Modality", style_table_cell_bold), Paragraph("Side-Scan Sonar (SSS)", style_table_cell)],
        [Paragraph("Model Architecture", style_table_cell_bold), Paragraph("YOLOv8s Custom-Trained SSS Detector (5-Class Taxonomy)", style_table_cell)],
        [Paragraph("Date of Compilation", style_table_cell_bold), Paragraph(datetime.now().strftime("%B %d, %Y"), style_table_cell)],
    ]
    t_meta = Table(meta_table_data, colWidths=[45 * mm, 125 * mm])
    t_meta.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, LINE_GRAY),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, LINE_GRAY),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ('BACKGROUND', (0, 0), (0, -1), CARD_BG),
    ]))
    story.append(t_meta)
    story.append(Spacer(1, 4 * mm))

    # Abstract
    story.append(Paragraph("<b>ABSTRACT</b>", ParagraphStyle('AbsTitle', fontName='Times-Bold', fontSize=9.5, leading=12, alignment=1, spaceAfter=4)))
    story.append(Paragraph(
        "Side-scan sonar (SSS) is an essential imaging modality for underwater inspection, seafloor mapping, and maritime hazard assessment. However, manual interpretation of large-area sonar surveys is labour-intensive and susceptible to interpreter fatigue caused by acoustic speckle noise, varying grazing angles, and complex natural seafloor topographies. This technical report presents the architecture, implementation, and empirical evaluation of <i>Marine Anomaly Intelligence</i> (SIH26057), an end-to-end software pipeline developed to automatically ingest SSS imagery, detect man-made marine debris and underwater objects, perform multi-signal evidence fusion, and generate localized survey dossiers. The system combines deep-learning object detection (YOLOv8s) with physics-based acoustic shadow analysis, autoencoder anomaly signals, and false-positive spatial filtering. On the untouched 700-image Drishti SSS test benchmark, the pipeline achieves an overall precision of 73.57%, recall of 67.85%, F1 score of 70.59%, and an mAP@50 of 68.78% (reaching 99.50% AP@50 on high-contrast submarine pipeline and ghost net targets). The complete system provides an operator inspection dashboard, geospatial referencing, human-in-the-loop review persistence, and multi-format export capabilities to support safe, efficient underwater survey workflows.",
        style_abstract_body
    ))
    story.append(Spacer(1, 3 * mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=LINE_GRAY, spaceAfter=8))

    # ---------------------------------------------------------
    # TABLE OF CONTENTS
    # ---------------------------------------------------------
    story.append(Paragraph("<b>TABLE OF CONTENTS</b>", style_h1))
    toc_data = [
        [Paragraph("1. Introduction", style_table_cell), Paragraph("9. Geospatial Reporting", style_table_cell)],
        [Paragraph("2. Problem Statement", style_table_cell), Paragraph("10. Dashboard and Operator Workflow", style_table_cell)],
        [Paragraph("3. Objectives", style_table_cell), Paragraph("11. Experimental Evaluation", style_table_cell)],
        [Paragraph("4. System Overview", style_table_cell), Paragraph("12. Results", style_table_cell)],
        [Paragraph("5. Side-Scan Sonar Data Processing", style_table_cell), Paragraph("13. Limitations", style_table_cell)],
        [Paragraph("6. AI-Based Object Detection", style_table_cell), Paragraph("14. Future Improvements", style_table_cell)],
        [Paragraph("7. Acoustic Shadow and Evidence Analysis", style_table_cell), Paragraph("15. Conclusion", style_table_cell)],
        [Paragraph("8. Confidence and False-Positive Filtering", style_table_cell), Paragraph("16. References", style_table_cell)],
    ]
    t_toc = Table(toc_data, colWidths=[85 * mm, 85 * mm])
    t_toc.setStyle(TableStyle([
        ('TOPPADDING', (0, 0), (-1, -1), 1.5),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 1.5),
        ('LEFTPADDING', (0, 0), (-1, -1), 0),
        ('RIGHTPADDING', (0, 0), (-1, -1), 0),
    ]))
    story.append(t_toc)
    story.append(Spacer(1, 4 * mm))

    # ---------------------------------------------------------
    # SECTION 1: INTRODUCTION
    # ---------------------------------------------------------
    story.append(Paragraph("1. Introduction", style_h1))
    story.append(Paragraph(
        "Marine debris, discarded fishing gear (ghost nets), shipwrecks, submerged pipelines, and discarded ordnance represent significant environmental hazards and navigational risks in coastal and deep-water domains. Side-scan sonar (SSS) serves as the primary acoustic sensor for wide-swath seabed reconnaissance. However, acoustic sonar imagery differs fundamentally from optical photography: acoustic propagation yields non-uniform ensonification, high speckle noise, intensity fall-off with range, and acoustic shadows that depend heavily on sonar altitude and seafloor topography. As survey areas expand to hundreds of square kilometres, manual visual inspection becomes a severe operational bottleneck. The SIH26057 project establishes an automated, auditable computational framework capable of assisting hydrographic surveyors by prioritizing anomalous acoustic returns, filtering natural seafloor clutter, and structuring findings for decision-makers.",
        style_body
    ))

    # ---------------------------------------------------------
    # SECTION 2: PROBLEM STATEMENT
    # ---------------------------------------------------------
    story.append(Paragraph("2. Problem Statement", style_h1))
    story.append(Paragraph(
        "Under problem statement <b>SIH26057</b>, the objective is to develop an automated software pipeline capable of ingesting raw side-scan sonar waterfall and tile imagery, detecting anthropogenic debris and target structures across designated object classes, distinguishing artificial objects from complex natural seabed textures (such as sand ripples, rock outcrops, and trenches), fusing acoustic shadow cues, attributing detection confidence and severity, and providing accessible operator inspection interfaces and structured technical dossiers.",
        style_body
    ))

    # ---------------------------------------------------------
    # SECTION 3: OBJECTIVES
    # ---------------------------------------------------------
    story.append(Paragraph("3. Objectives", style_h1))
    story.append(Paragraph(
        "The core technical objectives implemented in this system are summarized in Table 1.",
        style_body
    ))

    t1_data = [
        [Paragraph("ID", style_table_header), Paragraph("Objective Description", style_table_header), Paragraph("Implementation Scope", style_table_header)],
        [Paragraph("OBJ-1", style_table_cell_bold), Paragraph("Automated SSS Ingestion & Denoising", style_table_cell), Paragraph("Grayscale normalization, CLAHE contrast enhancement, dynamic range validation", style_table_cell)],
        [Paragraph("OBJ-2", style_table_cell_bold), Paragraph("Deep-Learning Object Localization", style_table_cell), Paragraph("YOLOv8s detector trained across 5 marine object classes", style_table_cell)],
        [Paragraph("OBJ-3", style_table_cell_bold), Paragraph("Acoustic Shadow Verification", style_table_cell), Paragraph("Morphological shadow extraction and object-to-shadow ratio verification", style_table_cell)],
        [Paragraph("OBJ-4", style_table_cell_bold), Paragraph("Multi-Signal Evidence Fusion", style_table_cell), Paragraph("Fused scoring combining YOLO confidence, shadow cues, and autoencoder loss", style_table_cell)],
        [Paragraph("OBJ-5", style_table_cell_bold), Paragraph("Operational Dashboard & Review", style_table_cell), Paragraph("Interactive UI, operator review feedback persistence in SQLite, multi-format export", style_table_cell)],
    ]
    t1 = Table(t1_data, colWidths=[18 * mm, 72 * mm, 80 * mm])
    t1.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, LINE_GRAY),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, LINE_GRAY),
        ('BACKGROUND', (0, 0), (-1, 0), CARD_BG),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t1)
    story.append(Paragraph("Table 1: SIH26057 core technical objectives and verified implementation scope.", style_caption))

    # ---------------------------------------------------------
    # SECTION 4: SYSTEM OVERVIEW & ARCHITECTURE
    # ---------------------------------------------------------
    story.append(Paragraph("4. System Overview & Architecture", style_h1))
    story.append(Paragraph(
        "The system follows a modular pipeline architecture wherein raw sonar frames pass sequentially through preprocessing, deep-learning feature extraction, physics-based shadow analysis, confidence fusion, and geospatial attribution before being committed to an ACID-compliant SQLite datastore and exposed via REST APIs. Figure 1 outlines the complete functional workflow.",
        style_body
    ))

    # Architecture Box Diagram Table
    arch_flow_data = [
        [Paragraph("<b>1. SSS Ingestion & Preprocessing</b><br/>CLAHE, Denoising, Quality Check", style_table_cell),
         Paragraph("<b>→</b>", style_table_cell_bold),
         Paragraph("<b>2. YOLOv8s Detection</b><br/>5-Class Bounding Box Proposal", style_table_cell),
         Paragraph("<b>→</b>", style_table_cell_bold),
         Paragraph("<b>3. Shadow & Anomaly Cues</b><br/>Morphological Shadow, AE Loss", style_table_cell)],
        [Paragraph("<b>4. Confidence Fusion</b><br/>Evidence Score & Severity (H/M/L)", style_table_cell),
         Paragraph("<b>→</b>", style_table_cell_bold),
         Paragraph("<b>5. Geospatial Attribution</b><br/>GPS / Metadata Source Tagging", style_table_cell),
         Paragraph("<b>→</b>", style_table_cell_bold),
         Paragraph("<b>6. Storage & Dossier Export</b><br/>SQLite DB, Dashboard, PDF / CSV", style_table_cell)]
    ]
    t_arch = Table(arch_flow_data, colWidths=[52 * mm, 6 * mm, 52 * mm, 6 * mm, 54 * mm])
    t_arch.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, NAVY),
        ('BACKGROUND', (0, 0), (-1, -1), CARD_BG),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_arch)
    story.append(Paragraph("Figure 1: End-to-end architecture of the SIH26057 underwater sonar analysis pipeline.", style_caption))

    # ---------------------------------------------------------
    # SECTION 5: SIDE-SCAN SONAR DATA PROCESSING
    # ---------------------------------------------------------
    story.append(Paragraph("5. Side-Scan Sonar Data Processing", style_h1))
    story.append(Paragraph(
        "Side-scan sonar records intensity backscatter as an acoustic transducer sweeps over the seabed. In the preprocessing module, raw images undergo contrast-limited adaptive histogram equalization (CLAHE) with a tile grid size of 8x8 and clip limit of 2.0 to balance acoustic intensity across the near-nadir blind zone and far-range falloff. In addition, bilateral filtering is selectively applied to smooth high-frequency speckle while preserving high-gradient object-to-shadow boundaries. Images are screened for quality metrics (dynamic range, sharpness, noise estimate) and missing-data dropout bands prior to inference.",
        style_body
    ))

    # ---------------------------------------------------------
    # SECTION 6: AI-BASED OBJECT DETECTION
    # ---------------------------------------------------------
    story.append(Paragraph("6. AI-Based Object Detection", style_h1))
    story.append(Paragraph(
        "The primary object detection module employs a custom-trained <b>YOLOv8s</b> neural network operating at 640x640 resolution with 3 multi-scale feature pyramid strides ([8, 16, 32]). The model taxonomy comprises 5 established target classes. Table 2 details the dataset splits used during development.",
        style_body
    ))

    t2_data = [
        [Paragraph("Dataset Split", style_table_header), Paragraph("Image Count", style_table_header), Paragraph("Annotation Count", style_table_header), Paragraph("Role in Evaluation", style_table_header)],
        [Paragraph("Drishti SSS Train", style_table_cell_bold), Paragraph("3,875 images", style_table_cell), Paragraph("5,210 targets", style_table_cell), Paragraph("Model training & weight optimization", style_table_cell)],
        [Paragraph("Drishti SSS Val", style_table_cell_bold), Paragraph("630 images", style_table_cell), Paragraph("854 targets", style_table_cell), Paragraph("Hyperparameter selection & checkpointing", style_table_cell)],
        [Paragraph("Drishti SSS Test (Untouched)", style_table_cell_bold), Paragraph("700 images", style_table_cell), Paragraph("948 targets", style_table_cell), Paragraph("Independent benchmark validation", style_table_cell)],
        [Paragraph("SubPipe HF/LF Swaths", style_table_cell_bold), Paragraph("2,066 images", style_table_cell), Paragraph("1,422 targets", style_table_cell), Paragraph("Cross-domain wide-aspect evaluation", style_table_cell)],
    ]
    t2 = Table(t2_data, colWidths=[40 * mm, 30 * mm, 35 * mm, 65 * mm])
    t2.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, LINE_GRAY),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, LINE_GRAY),
        ('BACKGROUND', (0, 0), (-1, 0), CARD_BG),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t2)
    story.append(Paragraph("Table 2: Dataset distribution across training, validation, test, and cross-domain evaluations.", style_caption))

    # Taxonomy Table
    t3_data = [
        [Paragraph("Class Index", style_table_header), Paragraph("Class Identifier", style_table_header), Paragraph("Display Name", style_table_header), Paragraph("Physical Characteristics in Sonar", style_table_header)],
        [Paragraph("0", style_table_cell_bold), Paragraph("crab_pot", style_table_cell), Paragraph("Crab Pot / Trap", style_table_cell), Paragraph("Small rectangular cage structure with compact shadow", style_table_cell)],
        [Paragraph("1", style_table_cell_bold), Paragraph("submarine_pipeline", style_table_cell), Paragraph("Submarine Pipeline", style_table_cell), Paragraph("Continuous linear high-intensity return with parallel shadow", style_table_cell)],
        [Paragraph("2", style_table_cell_bold), Paragraph("shipwreck", style_table_cell), Paragraph("Shipwreck / Hull", style_table_cell), Paragraph("Large irregular structural cluster with extensive acoustic shadow", style_table_cell)],
        [Paragraph("3", style_table_cell_bold), Paragraph("ghost_net", style_table_cell), Paragraph("Ghost Net / Gear", style_table_cell), Paragraph("Diffuse, undulating fibrous cluster draped over seabed", style_table_cell)],
        [Paragraph("4", style_table_cell_bold), Paragraph("mine_cylinder", style_table_cell), Paragraph("Mine / Cylinder", style_table_cell), Paragraph("Regular cylindrical / spherical body with sharp distinct shadow", style_table_cell)],
    ]
    t3 = Table(t3_data, colWidths=[20 * mm, 38 * mm, 38 * mm, 74 * mm])
    t3.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, LINE_GRAY),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, LINE_GRAY),
        ('BACKGROUND', (0, 0), (-1, 0), CARD_BG),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t3)
    story.append(Paragraph("Table 3: Established 5-class target taxonomy verified from model weights.", style_caption))

    # Training Curves & Confusion Matrix Figures
    fig_paths = [
            ROOT / 'docs' / 'evaluation' / 'curves' / 'results.png',
            ROOT / 'docs' / 'evaluation' / 'curves' / 'confusion_matrix.png'
    ]
    if fig_paths[0].exists() and fig_paths[1].exists():
        story.append(PageBreak())
        story.append(Paragraph("<b>Model Training Metrics and Confusion Matrix</b>", style_h2))
        img_w = 82 * mm
        img_h = 58 * mm
        rl_res = RLImage(str(fig_paths[0]), width=img_w, height=img_h)
        rl_cm = RLImage(str(fig_paths[1]), width=img_w, height=img_h)
        
        t_figs = Table([[rl_res, rl_cm]], colWidths=[85 * mm, 85 * mm])
        t_figs.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 0),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 0),
        ]))
        story.append(t_figs)
        story.append(Paragraph("Figure 2: (Left) YOLOv8s training loss convergence and mAP progression across 70 epochs. (Right) Normalised confusion matrix across evaluated acoustic target classes.", style_caption))

    # ---------------------------------------------------------
    # SECTION 7 & 8: SHADOW, ANOMALY, & EVIDENCE FUSION
    # ---------------------------------------------------------
    story.append(Paragraph("7. Acoustic Shadow and Evidence Analysis", style_h1))
    story.append(Paragraph(
        "In side-scan sonar physics, any positive relief object standing proud of the seafloor occludes sound rays, generating a characteristic dark acoustic shadow on the side opposite the transducer. The shadow analysis module crops candidate detection bounding boxes, estimates local seafloor intensity thresholds via Otsu adaptive thresholding, and computes the shadow area and object-to-shadow ratio. This metric provides a physical consistency check: an authentic seabed protrusion will produce a commensurate shadow, whereas seafloor specular reflectance artifacts will not. Shadow presence is treated as supporting evidence rather than absolute ground truth.",
        style_body
    ))

    story.append(Paragraph("8. Confidence Fusion & False-Positive Filtering", style_h1))
    story.append(Paragraph(
        "To prevent high-confidence false alarms on complex geological formations (e.g., boulder fields or sand dunes), the pipeline implements a multi-evidence fusion layer. The final evidence score combines: (1) raw YOLO classification confidence $C_{\\text{YOLO}}$, (2) shadow presence score $S_{\\text{shadow}}$, (3) auxiliary autoencoder reconstruction score $A_{\\text{anomaly}}$, and (4) geometric aspect ratio and minimum area consistency filters. Severity is categorized into HIGH (evidence $\\ge 0.50$ and high-risk hazard class), MEDIUM ($0.35 \\le \\text{evidence} < 0.50$), and LOW (evidence $< 0.35$).",
        style_body
    ))

    # ---------------------------------------------------------
    # SECTION 9 & 10: GEOSPATIAL & DASHBOARD WORKFLOW
    # ---------------------------------------------------------
    story.append(Paragraph("9. Geospatial Attribution & Metadata", style_h1))
    story.append(Paragraph(
        "Geospatial attribution strictly distinguishes between source origins to prevent misleading navigation logs. Every detection is assigned an explicit provenance tag in the database:",
        style_body
    ))

    t_geo_data = [
        [Paragraph("Source Tag", style_table_header), Paragraph("Definition", style_table_header), Paragraph("Operational Verification Rule", style_table_header)],
        [Paragraph("REAL_GPS", style_table_cell_bold), Paragraph("Direct NMEA/GPS Navigation Telemetry", style_table_cell), Paragraph("Requires verified hardware satellite fix in ping metadata", style_table_cell)],
        [Paragraph("SONAR_METADATA", style_table_cell_bold), Paragraph("Embedded XTF / JSF Sonar Header Coordinates", style_table_cell), Paragraph("Derived from acoustic ping coordinate packet", style_table_cell)],
        [Paragraph("MANUAL", style_table_cell_bold), Paragraph("Operator Survey Landmark Input", style_table_cell), Paragraph("Entered manually by hydrographic surveyor", style_table_cell)],
        [Paragraph("SIMULATED", style_table_cell_bold), Paragraph("Survey Grid Synthesized Coordinates", style_table_cell), Paragraph("Generated for offline demonstration grids", style_table_cell)],
        [Paragraph("UNAVAILABLE", style_table_cell_bold), Paragraph("No Coordinate Stream Present", style_table_cell), Paragraph("Assigned when sonar imagery lacks spatial metadata", style_table_cell)],
    ]
    t_geo = Table(t_geo_data, colWidths=[35 * mm, 65 * mm, 70 * mm])
    t_geo.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, LINE_GRAY),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, LINE_GRAY),
        ('BACKGROUND', (0, 0), (-1, 0), CARD_BG),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t_geo)
    story.append(Paragraph("Table 4: Geolocation provenance classification rules.", style_caption))

    story.append(Paragraph("10. Dashboard and Operator Workflow", style_h1))
    story.append(Paragraph(
        "The web-based inspection portal provides a human-in-the-loop verification environment. Operators can inspect recent waterfall swaths, inspect detected bounding boxes, review fused evidence scores, and submit audit feedback (CONFIRMED, FALSE ALARM, REJECTED) along with surveyor notes. All reviews are persisted directly to the backend SQLite datastore (`sih26057.db`), ensuring an immutable audit trail.",
        style_body
    ))

    # ---------------------------------------------------------
    # SECTION 11 & 12: EXPERIMENTAL EVALUATION & RESULTS
    # ---------------------------------------------------------
    story.append(Paragraph("11. Experimental Evaluation & Methodology", style_h1))
    story.append(Paragraph(
        "System evaluation was conducted using standard Pascal VOC / COCO object detection metrics at an IoU threshold of 0.50 (AP@50) and averaged across IoU thresholds from 0.50 to 0.95 (mAP@50-95). Evaluation was executed across three independent domains: (1) the untouched 700-image Drishti SSS test set, (2) the YOLOv8s 630-image validation split, and (3) the SubPipe HF/LF full-swath cross-domain benchmark. Table 5 and Table 6 document the verified empirical results.",
        style_body
    ))

    # Benchmark Test Metrics Table
    t5_data = [
        [Paragraph("Target Class", style_table_header), Paragraph("Precision", style_table_header), Paragraph("Recall", style_table_header), Paragraph("AP@50", style_table_header), Paragraph("AP@50-95", style_table_header)],
        [Paragraph("Submarine Pipeline", style_table_cell_bold), Paragraph("98.40%", style_table_cell), Paragraph("99.43%", style_table_cell), Paragraph("99.50%", style_table_cell_bold), Paragraph("79.21%", style_table_cell)],
        [Paragraph("Ghost Net", style_table_cell_bold), Paragraph("99.28%", style_table_cell), Paragraph("100.00%", style_table_cell), Paragraph("99.50%", style_table_cell_bold), Paragraph("89.20%", style_table_cell)],
        [Paragraph("Shipwreck", style_table_cell_bold), Paragraph("49.48%", style_table_cell), Paragraph("41.52%", style_table_cell), Paragraph("40.68%", style_table_cell_bold), Paragraph("22.23%", style_table_cell)],
        [Paragraph("Mine / Cylinder", style_table_cell_bold), Paragraph("47.13%", style_table_cell), Paragraph("30.44%", style_table_cell), Paragraph("35.45%", style_table_cell_bold), Paragraph("16.47%", style_table_cell)],
        [Paragraph("Crab Pot", style_table_cell_bold), Paragraph("N/A*", style_table_cell), Paragraph("N/A*", style_table_cell), Paragraph("N/A*", style_table_cell_bold), Paragraph("N/A*", style_table_cell)],
        [Paragraph("<b>Overall Benchmark Test (700 imgs)</b>", style_table_cell_bold), Paragraph("<b>73.57%</b>", style_table_cell_bold), Paragraph("<b>67.85%</b>", style_table_cell_bold), Paragraph("<b>68.78%</b>", style_table_cell_bold), Paragraph("<b>51.78%</b>", style_table_cell_bold)],
    ]
    t5 = Table(t5_data, colWidths=[45 * mm, 30 * mm, 30 * mm, 32 * mm, 33 * mm])
    t5.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, LINE_GRAY),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, LINE_GRAY),
        ('BACKGROUND', (0, 0), (-1, 0), CARD_BG),
        ('BACKGROUND', (0, -1), (-1, -1), CARD_BG),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t5)
    story.append(Paragraph("Table 5: Per-class performance on the untouched 700-image Drishti SSS test set. (*Crab pot ground-truth annotations were not present in the test split).", style_caption))

    # Latency Breakdown Table
    t6_data = [
        [Paragraph("Pipeline Component", style_table_header), Paragraph("Mean Execution Time", style_table_header), Paragraph("Hardware Platform", style_table_header), Paragraph("Throughput / Processing Rate", style_table_header)],
        [Paragraph("Preprocessing & CLAHE", style_table_cell_bold), Paragraph("7.80 ms", style_table_cell), Paragraph("Host CPU / OpenCV", style_table_cell), Paragraph("~128 frames / second", style_table_cell)],
        [Paragraph("YOLOv8s Object Detection", style_table_cell_bold), Paragraph("85.02 ms", style_table_cell), Paragraph("NVIDIA RTX 3050 GPU (CUDA)", style_table_cell), Paragraph("~12 frames / second", style_table_cell)],
        [Paragraph("Anomaly Autoencoder", style_table_cell_bold), Paragraph("2.30 ms", style_table_cell), Paragraph("NVIDIA RTX 3050 GPU (CUDA)", style_table_cell), Paragraph("~430 crops / second", style_table_cell)],
        [Paragraph("Acoustic Shadow Extraction", style_table_cell_bold), Paragraph("0.58 ms", style_table_cell), Paragraph("Host CPU / Morphological", style_table_cell), Paragraph("~1,700 crops / second", style_table_cell)],
        [Paragraph("Confidence Fusion & Filtering", style_table_cell_bold), Paragraph("0.36 ms", style_table_cell), Paragraph("Host CPU / Python", style_table_cell), Paragraph("~2,700 detections / second", style_table_cell)],
        [Paragraph("<b>Total Pipeline Latency (Mean)</b>", style_table_cell_bold), Paragraph("<b>99.55 ms</b>", style_table_cell_bold), Paragraph("<b>Combined System Pipeline</b>", style_table_cell_bold), Paragraph("<b>~10 frames / second (Real-Time)</b>", style_table_cell_bold)],
    ]
    t6 = Table(t6_data, colWidths=[48 * mm, 34 * mm, 48 * mm, 40 * mm])
    t6.setStyle(TableStyle([
        ('BOX', (0, 0), (-1, -1), 0.5, LINE_GRAY),
        ('INNERGRID', (0, 0), (-1, -1), 0.5, LINE_GRAY),
        ('BACKGROUND', (0, 0), (-1, 0), CARD_BG),
        ('BACKGROUND', (0, -1), (-1, -1), CARD_BG),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]))
    story.append(t6)
    story.append(Paragraph("Table 6: Stage-wise latency benchmark measurements on NVIDIA RTX 3050 Laptop GPU.", style_caption))

    # ---------------------------------------------------------
    # SECTION 13 & 14: LIMITATIONS & FUTURE IMPROVEMENTS
    # ---------------------------------------------------------
    story.append(Paragraph("13. System Limitations", style_h1))
    story.append(Paragraph(
        "A scientifically rigorous evaluation reveals several operational limitations that must be addressed before deployment on operational autonomous underwater vehicles (AUVs):",
        style_body
    ))
    story.append(Paragraph("• <b>SubPipe Aspect Ratio Mismatch:</b> Direct, untiled inference on raw SubPipe swaths (5000x500 px and 2500x500 px) produced 0 true positives due to extreme 10:1 and 5:1 aspect ratios, demonstrating that wide waterfall imagery strictly requires aspect-ratio-aware sliding window tiling.", style_body))
    story.append(Paragraph("• <b>Shipwreck & Cylinder Sensitivity:</b> Shipwreck AP@50 (40.68%) and mine cylinder AP@50 (35.45%) reflect high intra-class geometric variability and acoustic reverberation challenges in complex rocky seabeds.", style_body))
    story.append(Paragraph("• <b>Auxiliary Autoencoder Discriminability:</b> The auxiliary autoencoder produces low anomaly discrimination on subtle seabed textures and is currently utilized only as a secondary auxiliary metric rather than a standalone classifier.", style_body))
    story.append(Paragraph("• <b>Crab Pot Test Support:</b> While present in the model taxonomy (Index 0), the test split lacked annotated crab pot instances for independent scoring.", style_body))

    story.append(Paragraph("14. Future Improvements", style_h1))
    story.append(Paragraph(
        "Recommended future improvements include: (1) implementing automated multi-scale sliding window tiling for extreme aspect-ratio sonar swaths, (2) integrating real XTF/JSF sonar packet parsers for direct navigation fix extraction, (3) expanding training datasets with real offshore ghost-gear recordings, and (4) deploying ONNX / TensorRT quantised models for embedded edge computing on AUV payloads.",
        style_body
    ))

    # ---------------------------------------------------------
    # SECTION 15 & 16: CONCLUSION & REFERENCES
    # ---------------------------------------------------------
    story.append(Paragraph("15. Conclusion", style_h1))
    story.append(Paragraph(
        "The SIH26057 Marine Anomaly Intelligence project successfully demonstrates a fully functional prototype for automated side-scan sonar object detection, multi-signal evidence fusion, geospatial attribution, and surveyor reporting. By integrating deep-learning object localization with acoustic shadow verification and interactive operator audit logging, the system provides a robust technological foundation for automated marine environmental protection, infrastructure monitoring, and seabed reconnaissance.",
        style_body
    ))

    story.append(Paragraph("16. References", style_h1))
    story.append(Paragraph("[1] Ministry of Earth Sciences & National Institute of Ocean Technology, <i>Smart India Hackathon 2026 Problem Statement SIH26057: AI for Marine Debris Detection</i>, 2026.", style_body))
    story.append(Paragraph("[2] Drishti SSS Acoustic Sonar Benchmark Dataset, <i>Multi-Class Underwater Object Detection in Side-Scan Sonar Imagery</i>, 2025.", style_body))
    story.append(Paragraph("[3] Jocher, G., et al., <i>Ultralytics YOLOv8 Architecture and Real-Time Object Detection Framework</i>, 2023.", style_body))
    story.append(Paragraph("[4] SubPipe Sonar Inspection Dataset, <i>Autonomous Submarine Pipeline Inspection Using Acoustic Side-Scan Sonar</i>, 2024.", style_body))

    # Build document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"PDF generated successfully at: {output_path}")


if __name__ == "__main__":
    out_file = str(ROOT / 'docs' / 'reports' / 'SIH26057_Technical_Report.pdf')
    generate_pdf(out_file)
    
    # Also save in sih26057-standalone/reports
    standalone_reports = ROOT / 'docs' / 'reports'
    standalone_reports.mkdir(parents=True, exist_ok=True)
    out_file_standalone = str(standalone_reports / "SIH26057_Technical_Report.pdf")
    generate_pdf(out_file_standalone)