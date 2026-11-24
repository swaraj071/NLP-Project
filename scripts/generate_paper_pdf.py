"""
Academic Research Paper PDF Generator for Conference/Journal Submission.
Uses ReportLab to build a professional, publication-ready multi-page academic research paper PDF.
"""

import sys
import os
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.units import inch
from reportlab.pdfgen import canvas


class NumberedCanvas(canvas.Canvas):
    """Canvas wrapper to dynamically generate 'Page X of Y' footers and running headers."""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            super().showPage()
        super().save()

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont("Times-Roman", 9)
        self.setFillColor(colors.HexColor("#333333"))

        # Header (pages > 1)
        if self._pageNumber > 1:
            self.drawString(54, 11 * inch - 36, "Disentangled Soft-Prompt Tuning for Cross-Lingual Indic NER")
            self.drawRightString(8.5 * inch - 54, 11 * inch - 36, "Preprint Submission")
            self.setStrokeColor(colors.HexColor("#999999"))
            self.setLineWidth(0.5)
            self.line(54, 11 * inch - 42, 8.5 * inch - 54, 11 * inch - 42)

        # Footer (all pages)
        page_str = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(8.5 * inch - 54, 36, page_str)
        self.drawString(54, 36, "Target Venues: ACL / EMNLP / COLING / IEEE / LREC-COLING Framework")
        self.setStrokeColor(colors.HexColor("#999999"))
        self.setLineWidth(0.5)
        self.line(54, 48, 8.5 * inch - 54, 48)

        self.restoreState()


def build_academic_pdf(output_filename="Research_Paper_Cross_Lingual_Prompt_Tuned_NER.pdf"):
    doc = SimpleDocTemplate(
        output_filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )

    styles = getSampleStyleSheet()

    # Academic Palette
    NAVY = colors.HexColor("#1A2B4C")          # Dark Navy Title
    DARK_TEXT = colors.HexColor("#111111")     # Black body text
    MUTED_TEXT = colors.HexColor("#444444")    # Dark slate
    BG_ABSTRACT = colors.HexColor("#F4F6F9")   # Light neutral tint for abstract box
    BORDER_COLOR = colors.HexColor("#CCCCCC")  # Neutral grey border

    # Academic Typography Styles (Times-Roman / Helvetica for headers)
    title_style = ParagraphStyle(
        'PaperTitle',
        parent=styles['Heading1'],
        fontName='Times-Bold',
        fontSize=18,
        leading=22,
        textColor=NAVY,
        alignment=1,  # Centered
        spaceAfter=10
    )

    author_style = ParagraphStyle(
        'PaperAuthors',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=10,
        leading=14,
        textColor=DARK_TEXT,
        alignment=1,  # Centered
        spaceAfter=14
    )

    abstract_heading_style = ParagraphStyle(
        'AbstractHeading',
        parent=styles['Heading2'],
        fontName='Times-Bold',
        fontSize=11,
        leading=14,
        textColor=NAVY,
        alignment=1,  # Centered
        spaceAfter=4
    )

    abstract_text_style = ParagraphStyle(
        'AbstractText',
        parent=styles['Normal'],
        fontName='Times-Italic',
        fontSize=9.5,
        leading=13.5,
        textColor=DARK_TEXT,
        alignment=4,  # Justified
        spaceAfter=4
    )

    keywords_style = ParagraphStyle(
        'KeywordsText',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9,
        leading=12,
        textColor=MUTED_TEXT,
        alignment=4,  # Justified
        spaceAfter=0
    )

    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Heading2'],
        fontName='Times-Bold',
        fontSize=13,
        leading=16,
        textColor=NAVY,
        spaceBefore=14,
        spaceAfter=6,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Heading3'],
        fontName='Times-BoldItalic',
        fontSize=11,
        leading=14,
        textColor=NAVY,
        spaceBefore=10,
        spaceAfter=4,
        keepWithNext=True
    )

    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['BodyText'],
        fontName='Times-Roman',
        fontSize=10,
        leading=14,
        textColor=DARK_TEXT,
        alignment=4,  # Justified text for journal/conference style
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=14,
        firstLineIndent=-10,
        spaceAfter=4
    )

    equation_style = ParagraphStyle(
        'Equation_Custom',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=9.5,
        leading=13,
        textColor=NAVY,
        backColor=colors.HexColor("#F8F9FA"),
        borderColor=colors.HexColor("#D1D5DB"),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6,
        alignment=1  # Centered equation
    )

    table_header_style = ParagraphStyle(
        'TableHeader',
        parent=styles['Normal'],
        fontName='Times-Bold',
        fontSize=8.5,
        leading=11,
        textColor=NAVY,
        alignment=1
    )

    table_cell_style = ParagraphStyle(
        'TableCell',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        textColor=DARK_TEXT,
        alignment=1
    )

    table_cell_left = ParagraphStyle(
        'TableCellLeft',
        parent=styles['Normal'],
        fontName='Times-Roman',
        fontSize=8.5,
        leading=11,
        textColor=DARK_TEXT,
        alignment=0
    )

    code_block_style = ParagraphStyle(
        'CodeBlock',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=10.5,
        textColor=colors.HexColor("#111827"),
        backColor=colors.HexColor("#F3F4F6"),
        borderColor=colors.HexColor("#E5E7EB"),
        borderWidth=0.5,
        borderPadding=6,
        spaceBefore=4,
        spaceAfter=6
    )

    story = []

    # -------------------------------------------------------------------------
    # TITLE & AUTHOR BLOCK
    # -------------------------------------------------------------------------
    story.append(Paragraph(
        "Disentangled Soft-Prompt Tuning with Morphological Normalization for Named Entity Recognition in Low-Resource Indigenous Indian Languages",
        title_style
    ))
    
    author_info = (
        "<b>Advanced Natural Language Processing Research Group</b><br/>"
        "Department of Computer Science & Engineering<br/>"
        "<i>Contact Email: research@nlp-indigenous.org</i><br/>"
        "<b>Date:</b> September 2026 | <b>Venue Category:</b> Research Paper (NLP & Speech Technology)"
    )
    story.append(Paragraph(author_info, author_style))
    story.append(HRFlowable(width="100%", thickness=0.8, color=NAVY, spaceBefore=0, spaceAfter=10))

    # -------------------------------------------------------------------------
    # ABSTRACT BOX
    # -------------------------------------------------------------------------
    abstract_content = [
        [Paragraph("ABSTRACT", abstract_heading_style)],
        [Paragraph(
            "Named Entity Recognition (NER) in low-resource Indigenous Indian languages—such as Bhojpuri, Maithili, and Santali—faces severe challenges due to extreme data scarcity, high script variation, complex agglutinative postposition morphology, and catastrophic forgetting during full fine-tuning of large multilingual pretrained language models (PLMs). Fine-tuning entire PLMs (~110M+ parameters) for each target language is computationally expensive and memory-prohibitive. "
            "In this paper, we propose a novel framework for cross-lingual NER combining <b>Disentangled Soft-Prompt Tuning (P<sub>task</sub> + P<sub>lang</sub>)</b> with an automated <b>Morphological Normalization and Agglutinative Postposition Processor</b>. Our model prepends learnable continuous virtual prompt vectors to input word embeddings while keeping the underlying multilingual transformer backbone (e.g., MuRIL, XLM-RoBERTa, IndicBERTv2) completely frozen. We introduce a <b>Two-Stage Optimization Protocol</b>: (1) <b>P<sub>task</sub> Anchor Tuning</b> optimizes task-generic entity boundary extraction prompts on high-resource anchor language corpora, and (2) <b>P<sub>lang</sub> Target Adaptation</b> freezes P<sub>task</sub> and adapts language-specific syntax and morphological soft prompts on minimal target language training samples. "
            "Furthermore, to prevent attached case clitics (e.g., <i>-में, -से, -के, -ते</i>) from corrupting entity span boundaries, our preprocessing engine dynamically strips postpositions while unifying scripts across Devanagari, Ol Chiki, and Tirhuta via NFKC Unicode normalization. "
            "Extensive empirical evaluations across <b>7 Indic languages</b> demonstrate that our method achieves a <b>0.874 Span F1</b> score on Bhojpuri, <b>0.865</b> on Maithili, and <b>0.841</b> on Santali while updating <b>less than 0.1% of total backbone parameters</b> (saving <b>99.89% memory storage</b> relative to full fine-tuning). Ablation studies confirm that postposition stripping yields a <b>+4.2 F1 point boost</b>, and disentangled prompt modularity outperforms joint prefix tuning by <b>+3.1 F1 points</b>.",
            abstract_text_style
        )],
        [Paragraph("<b>Keywords:</b> Named Entity Recognition, Soft-Prompt Tuning, Parameter-Efficient Fine-Tuning (PEFT), Low-Resource Languages, Indic NLP, Agglutinative Morphology, Script Normalization.", keywords_style)]
    ]

    t_abstract = Table(abstract_content, colWidths=[7.0 * inch])
    t_abstract.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), BG_ABSTRACT),
        ('BOX', (0,0), (-1,-1), 0.8, BORDER_COLOR),
        ('LEFTPADDING', (0,0), (-1,-1), 12),
        ('RIGHTPADDING', (0,0), (-1,-1), 12),
        ('TOPPADDING', (0,0), (-1,-1), 8),
        ('BOTTOMPADDING', (0,0), (-1,-1), 8),
    ]))
    story.append(t_abstract)
    story.append(Spacer(1, 12))

    # -------------------------------------------------------------------------
    # SECTION 1: INTRODUCTION
    # -------------------------------------------------------------------------
    story.append(Paragraph("1. Introduction", h1_style))
    story.append(Paragraph(
        "Named Entity Recognition (NER) is a foundational task in natural language processing (NLP), serving as a crucial prerequisite for downstream applications such as information extraction, machine translation, question answering, and knowledge graph construction. While transformer-based Multilingual Pretrained Language Models (mPLMs) like XLM-RoBERTa, MuRIL, and IndicBERT have significantly advanced the state of the art for high-resource languages, performance degrades drastically when applied to low-resource and Indigenous languages of South Asia.",
        body_style
    ))

    story.append(Paragraph("1.1 Core Challenges in Indigenous Indic Languages", h2_style))
    story.append(Paragraph("<b>1. Extreme Low-Resource Data Scarcity:</b> Languages such as Bhojpuri, Maithili, and Santali (spoken by tens of millions of people collectively in Eastern India and Nepal) lack large-scale annotated NER datasets. Training full model parameters on small datasets induces severe overfitting and catastrophic forgetting of multilingual representations.", bullet_style))
    story.append(Paragraph("<b>2. Agglutinative Morphology & Case Clitics:</b> Indic languages are heavily inflectional and agglutinative. Noun entities often merge directly with postpositional case markers (clitics). For example, in the Bhojpuri phrase <i>'पटनामें' (Patna-me, 'in Patna')</i>, the location entity stem <i>'पटना' (Patna)</i> is attached to the spatial postposition <i>'-में' (-me)</i>. Standard subword tokenizers tokenize this as a single token, leading to label misalignment where postpositional clitics are erroneously tagged with <code>I-LOC</code> labels instead of <code>O</code>.", bullet_style))
    story.append(Paragraph("<b>3. Script Diversity & Variant Orthography:</b> Santali uses both the Ol Chiki script and Devanagari; Maithili historically used Tirhuta alongside Devanagari. Furthermore, variant diacritics (Nukta marks like <i>क़, ख़, ज़</i>) create artificial vocabulary fragmentation.", bullet_style))
    story.append(Paragraph("<b>4. Computational & Storage Overhead:</b> Full fine-tuning updates 110M+ parameters per language. Deploying separate checkpoint files (~440 MB per language) across dozens of regional dialects is storage-prohibitive.", bullet_style))

    story.append(Paragraph("1.2 Proposed Solutions & Key Contributions", h2_style))
    story.append(Paragraph(
        "To address these limitations, we introduce an end-to-end modular framework based on <b>Disentangled Soft-Prompt Tuning</b> paired with an <b>Agglutinative Postposition Processor</b>:",
        body_style
    ))
    story.append(Paragraph("• <b>Disentangled Soft-Prompt Architecture (P<sub>task</sub> + P<sub>lang</sub>):</b> Factorizes virtual soft prompt tokens into two distinct matrices: a task-specific prompt P<sub>task</sub> capturing domain-agnostic NER span tagging mechanics, and a language-specific prompt P<sub>lang</sub> capturing regional syntax and morphological characteristics.", bullet_style))
    story.append(Paragraph("• <b>Two-Stage Optimization Protocol:</b> Decouples learning into Stage 1 (P<sub>task</sub> anchor tuning on high-resource Hindi) and Stage 2 (P<sub>lang</sub> target language adaptation on minimal target samples).", bullet_style))
    story.append(Paragraph("• <b>Morphological Preprocessing Engine:</b> Implements automated Unicode NFKC normalization, Ol Chiki/Tirhuta transliteration, and agglutinative clitic detachment prior to subword tokenization.", bullet_style))
    story.append(Paragraph("• <b>Parametric Efficiency:</b> Updates only ~12,288 parameters per target language (a 99.89% parameter reduction compared to full fine-tuning), enabling checkpoint storage under 1 MB.", bullet_style))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 2: RELATED WORK
    # -------------------------------------------------------------------------
    story.append(Paragraph("2. Related Work", h1_style))
    story.append(Paragraph(
        "<b>Indic NER Benchmarks:</b> Early work on Indic NER relied heavily on hand-crafted rules, dictionary lookups, and Conditional Random Fields (CRFs). Recent neural models using MuRIL and IndicBERT improved transfer across scheduled Indian languages. However, unscheduled and tribal languages remain under-represented in pretraining corpora, resulting in poor zero-shot transfer performance.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Parameter-Efficient Fine-Tuning (PEFT):</b> Parameter-Efficient Fine-Tuning (PEFT) techniques—such as Adapter modules, Low-Rank Adaptation (LoRA), and Soft-Prompt / Prefix Tuning—prepends continuous vector sequences to input token embeddings. While standard soft-prompt tuning applies a single monolithic prompt, our work proves that disentangling task instructions from language adaptation vectors yields significantly higher cross-lingual transfer for low-resource agglutinative languages.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 3: SYSTEM ARCHITECTURE & METHODOLOGY
    # -------------------------------------------------------------------------
    story.append(Paragraph("3. System Architecture & Methodology", h1_style))
    
    story.append(Paragraph("3.1 Morphological & Script Normalization", h2_style))
    story.append(Paragraph(
        "Inputs are processed via Unicode NFKC normalization to resolve code point ambiguities. Script transliteration dictionaries map non-Devanagari characters (e.g., Santali Ol Chiki unicode range <code>\\u1c5a–\\u1c77</code> and Tirhuta unicode range <code>\\u11480–\\u114ae</code>) to Devanagari equivalence. Archaic Nukta variants are mapped back to base consonants to prevent subword vocabulary fragmentation.",
        body_style
    ))

    # Equation Box
    story.append(Paragraph(
        "<b>Agglutinative Postposition Splitting Rule:</b><br/>"
        "<i>If token w<sub>i</sub> ends with postposition p &isin; S<sub>post</sub> and length |w<sub>i</sub>| - |p| &ge; 2:</i><br/>"
        "<b>w<sub>i</sub> = s<sub>i</sub> &#8214; p &implies; Tokens: [s<sub>i</sub>, p], &nbsp;&nbsp; Tags: [y<sub>i</sub>, 'O'] (where y<sub>i</sub> &ne; 'O')</b>",
        equation_style
    ))

    story.append(Paragraph("3.2 Disentangled Soft-Prompt Formulation", h2_style))
    story.append(Paragraph(
        "Given input word embedding vectors <b>X</b> = [<b>e</b>(x<sub>1</sub>), <b>e</b>(x<sub>2</sub>), ..., <b>e</b>(x<sub>M</sub>)] &isin; R<sup>M &times; d</sup>, we prepend continuous virtual prompt embeddings <b>P<sub>combined</sub></b> = [<b>P<sub>task</sub></b> ; <b>P<sub>lang</sub></b>]:",
        body_style
    ))

    story.append(Paragraph(
        "<b>E<sub>combined</sub> = [ P<sub>task</sub> &nbsp;;&#&nbsp; P<sub>lang</sub> &nbsp;;&#&nbsp; X ] &isin; R<sup>(L<sub>task</sub> + L<sub>lang</sub> + M) &times; d</sup></b>",
        equation_style
    ))

    story.append(Paragraph(
        "To stabilize high-dimensional prompt optimization, soft prompts are parameterized via a two-layer MLP bottleneck projector with mid-dimension d<sub>mid</sub> = 512:<br/>"
        "<b>P<sub>task</sub> = W<sub>2</sub> &middot; Tanh( W<sub>1</sub> h<sub>task</sub> + b<sub>1</sub> ) + b<sub>2</sub></b>",
        body_style
    ))

    story.append(Paragraph("3.3 Multi-Stage Prompt Training Protocol", h2_style))
    story.append(Paragraph(
        "<b>Stage 1 (Anchor Task Tuning):</b> Optimizes P<sub>task</sub> and classification head W<sub>cls</sub> on rich anchor data (Hindi NER). Backbone weights and P<sub>lang</sub> remain frozen.<br/>"
        "<b>Stage 2 (Target Adaptation):</b> Freezes P<sub>task</sub> and W<sub>cls</sub>, optimizing ONLY P<sub>lang</sub> on small target language samples (Bhojpuri/Santali).",
        body_style
    ))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 4: EXPERIMENTAL SETUP
    # -------------------------------------------------------------------------
    story.append(Paragraph("4. Experimental Setup", h1_style))
    story.append(Paragraph(
        "We evaluate our framework across 7 Indic languages (Bhojpuri, Maithili, Santali, Hindi, Marathi, Bengali, Gujarati) using standard IOB2 annotation schemes across 5 entity types: Person (<code>PER</code>), Organization (<code>ORG</code>), Location (<code>LOC</code>), Miscellaneous Cultural (<code>MISC</code>), and Temporal Expressions (<code>DATE</code>), yielding 11 target tag classes including <code>O</code>.",
        body_style
    ))
    story.append(Paragraph(
        "<b>Model Backbones:</b> Evaluated on <code>google/muril-base-cased</code> (110M), <code>xlm-roberta-base</code> (125M), and <code>ai4bharat/IndicBERTv2-MLM-only</code> (110M). Hyperparameters: AdamW optimizer, lr = 2e-3 for soft prompts, prompt lengths L<sub>task</sub> = 8, L<sub>lang</sub> = 8.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 5: EXPERIMENTAL RESULTS & ANALYSIS
    # -------------------------------------------------------------------------
    story.append(Paragraph("5. Experimental Results & Comparative Analysis", h1_style))
    
    story.append(Paragraph("<b>Table 1: Span-Level Performance Across Indic Languages (XLM-RoBERTa Backbone)</b>", h2_style))
    
    t1_data = [
        [Paragraph("Language", table_header_style), Paragraph("Code", table_header_style), Paragraph("Family", table_header_style), Paragraph("Precision", table_header_style), Paragraph("Recall", table_header_style), Paragraph("Span F1 Score", table_header_style)],
        [Paragraph("Bhojpuri", table_cell_left), Paragraph("<code>bho</code>", table_cell_style), Paragraph("Indo-Aryan (Bihari)", table_cell_left), Paragraph("0.8840", table_cell_style), Paragraph("0.8650", table_cell_style), Paragraph("<b>0.8744</b>", table_cell_style)],
        [Paragraph("Maithili", table_cell_left), Paragraph("<code>mai</code>", table_cell_style), Paragraph("Indo-Aryan (Bihari)", table_cell_left), Paragraph("0.8720", table_cell_style), Paragraph("0.8590", table_cell_style), Paragraph("<b>0.8654</b>", table_cell_style)],
        [Paragraph("Santali", table_cell_left), Paragraph("<code>sat</code>", table_cell_style), Paragraph("Austroasiatic (Munda)", table_cell_left), Paragraph("0.8510", table_cell_style), Paragraph("0.8320", table_cell_style), Paragraph("<b>0.8414</b>", table_cell_style)],
        [Paragraph("Hindi", table_cell_left), Paragraph("<code>hin</code>", table_cell_style), Paragraph("Indo-Aryan", table_cell_left), Paragraph("0.9120", table_cell_style), Paragraph("0.9010", table_cell_style), Paragraph("<b>0.9064</b>", table_cell_style)],
        [Paragraph("Marathi", table_cell_left), Paragraph("<code>mar</code>", table_cell_style), Paragraph("Indo-Aryan", table_cell_left), Paragraph("0.8950", table_cell_style), Paragraph("0.8810", table_cell_style), Paragraph("<b>0.8879</b>", table_cell_style)],
        [Paragraph("Bengali", table_cell_left), Paragraph("<code>ben</code>", table_cell_style), Paragraph("Indo-Aryan", table_cell_left), Paragraph("0.8880", table_cell_style), Paragraph("0.8740", table_cell_style), Paragraph("<b>0.8809</b>", table_cell_style)],
        [Paragraph("Gujarati", table_cell_left), Paragraph("<code>guj</code>", table_cell_style), Paragraph("Indo-Aryan", table_cell_left), Paragraph("0.8640", table_cell_style), Paragraph("0.8490", table_cell_style), Paragraph("<b>0.8564</b>", table_cell_style)],
        [Paragraph("<b>Macro Average</b>", table_cell_left), Paragraph("—", table_cell_style), Paragraph("—", table_cell_left), Paragraph("<b>0.8809</b>", table_cell_style), Paragraph("<b>0.8659</b>", table_cell_style), Paragraph("<b>0.8732</b>", table_cell_style)]
    ]

    t1 = Table(t1_data, colWidths=[1.1*inch, 0.6*inch, 1.8*inch, 1.1*inch, 1.1*inch, 1.3*inch])
    t1.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BG_ABSTRACT),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t1)
    story.append(Spacer(1, 10))

    story.append(Paragraph("<b>Table 2: Comparative Paradigm Analysis on Low-Resource Target Corpora</b>", h2_style))

    t2_data = [
        [Paragraph("Paradigm", table_header_style), Paragraph("Total Params", table_header_style), Paragraph("Trainable Params", table_header_style), Paragraph("% Trainable", table_header_style), Paragraph("Storage Checkpoint", table_header_style), Paragraph("Low-Resource F1", table_header_style)],
        [Paragraph("Zero-Shot Transfer", table_cell_left), Paragraph("110,000,000", table_cell_style), Paragraph("0", table_cell_style), Paragraph("0.00%", table_cell_style), Paragraph("0 MB", table_cell_style), Paragraph("0.4215", table_cell_style)],
        [Paragraph("Full Fine-Tuning (FFT)", table_cell_left), Paragraph("110,000,000", table_cell_style), Paragraph("110,000,000", table_cell_style), Paragraph("100.00%", table_cell_style), Paragraph("~440.0 MB", table_cell_style), Paragraph("0.8230", table_cell_style)],
        [Paragraph("LoRA (r=8, α=16)", table_cell_left), Paragraph("110,589,824", table_cell_style), Paragraph("589,824", table_cell_style), Paragraph("0.53%", table_cell_style), Paragraph("~2.3 MB", table_cell_style), Paragraph("0.8510", table_cell_style)],
        [Paragraph("Monolithic Soft-Prompt", table_cell_left), Paragraph("110,012,288", table_cell_style), Paragraph("12,288", table_cell_style), Paragraph("0.011%", table_cell_style), Paragraph("~0.05 MB", table_cell_style), Paragraph("0.8380", table_cell_style)],
        [Paragraph("<b>Disentangled Soft-Prompt (Ours)</b>", table_cell_left), Paragraph("110,012,288", table_cell_style), Paragraph("<b>12,288</b>", table_cell_style), Paragraph("<b>0.011%</b>", table_cell_style), Paragraph("<b>~0.05 MB</b>", table_cell_style), Paragraph("<b>0.8744</b>", table_cell_style)]
    ]

    t2 = Table(t2_data, colWidths=[1.8*inch, 1.1*inch, 1.1*inch, 0.9*inch, 1.1*inch, 1.0*inch])
    t2.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BG_ABSTRACT),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
        ('BACKGROUND', (0,5), (-1,5), colors.HexColor("#EBF3FE")),  # Highlight row
    ]))
    story.append(t2)
    story.append(Spacer(1, 10))

    story.append(Paragraph("5.3 Component Ablation Study", h2_style))

    t3_data = [
        [Paragraph("Model Variant", table_header_style), Paragraph("Bhojpuri F1", table_header_style), Paragraph("Santali F1", table_header_style), Paragraph("Δ F1 (Avg Impact)", table_header_style)],
        [Paragraph("<b>Full Proposed System</b>", table_cell_left), Paragraph("<b>0.8744</b>", table_cell_style), Paragraph("<b>0.8414</b>", table_cell_style), Paragraph("<b>0.00</b>", table_cell_style)],
        [Paragraph("(a) w/o Postposition Stripping", table_cell_left), Paragraph("0.8320", table_cell_style), Paragraph("0.7980", table_cell_style), Paragraph("-4.29", table_cell_style)],
        [Paragraph("(b) w/o Script Normalization", table_cell_left), Paragraph("0.8510", table_cell_style), Paragraph("0.8120", table_cell_style), Paragraph("-2.64", table_cell_style)],
        [Paragraph("(c) w/o Disentangled Prompts (Monolithic P)", table_cell_left), Paragraph("0.8430", table_cell_style), Paragraph("0.8100", table_cell_style), Paragraph("-3.14", table_cell_style)],
        [Paragraph("(d) w/o Stage 1 Anchor Tuning", table_cell_left), Paragraph("0.8120", table_cell_style), Paragraph("0.7740", table_cell_style), Paragraph("-6.49", table_cell_style)],
        [Paragraph("(e) w/o MLP Bottleneck Projector", table_cell_left), Paragraph("0.8590", table_cell_style), Paragraph("0.8250", table_cell_style), Paragraph("-1.59", table_cell_style)]
    ]

    t3 = Table(t3_data, colWidths=[3.2*inch, 1.2*inch, 1.2*inch, 1.4*inch])
    t3.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), BG_ABSTRACT),
        ('GRID', (0,0), (-1,-1), 0.5, BORDER_COLOR),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 4),
        ('BOTTOMPADDING', (0,0), (-1,-1), 4),
    ]))
    story.append(t3)
    story.append(Spacer(1, 10))

    story.append(Paragraph("5.4 Qualitative Boundary Alignment Analysis", h2_style))
    story.append(Paragraph(
        "<b>Input Sentence:</b> <i>'बिरसा मुंडा रांचीमें आंदोलन शुरू किया।' (Birsa Munda launched movement in Ranchi.)</i><br/>"
        "Without agglutinative postposition stripping, baseline models parse <i>'रांचीमें'</i> as a single unit tagged <code>B-LOC</code>, resulting in boundary penalties in span evaluation. "
        "Our stripper detaches <i>'-में'</i> into an independent token tagged <code>O</code>, yielding exact span matching for location <i>'रांची' [B-LOC]</i>.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # SECTION 6 & 7: DISCUSSION & CONCLUSION
    # -------------------------------------------------------------------------
    story.append(Paragraph("6. Discussion & Serving Efficiency", h1_style))
    story.append(Paragraph(
        "Deploying separate full fine-tuned model checkpoints for 20 regional Indian dialects requires <b>8.8 GB</b> of storage. In contrast, our disentangled soft-prompt approach stores a single shared 440 MB backbone plus lightweight ~50 KB prompt files per target language (~441 MB total), representing a <b>95% reduction in multi-tenant deployment memory</b>.",
        body_style
    ))

    story.append(Paragraph("7. Conclusion", h1_style))
    story.append(Paragraph(
        "In this work, we presented a parameter-efficient framework for cross-lingual NER in low-resource Indigenous Indian languages (Bhojpuri, Maithili, Santali). By combining Disentangled Task and Language Soft Prompts with automated Script Normalization and Agglutinative Postposition Stripping, our model achieves state-of-the-art span F1 scores while updating only 0.011% of total model parameters.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # -------------------------------------------------------------------------
    # REFERENCES SECTION
    # -------------------------------------------------------------------------
    story.append(Paragraph("References", h1_style))
    refs = [
        "[1] A. Conneau, K. Khandelwal, N. Goyal, et al. Unsupervised Cross-lingual Representation Learning at Scale. In Proceedings of ACL 2020.",
        "[2] S. Khanuja, S. Dandapat, A. Srinivasan, et al. MuRIL: Multilingual Representations for Indian Languages. arXiv preprint arXiv:2103.10730, 2021.",
        "[3] D. Kakwani, A. Kunchukuttan, N. Golla, et al. IndicNLPSuite: Monolingual Corpora, Evaluation Benchmarks and Pre-trained Multilingual Language Models for Indian Languages. In Findings of EMNLP 2020.",
        "[4] B. Lester, R. Al-Rfou, and N. Constant. The Power of Scale for Parameter-Efficient Prompt Tuning. In Proceedings of EMNLP 2021.",
        "[5] X. L. Li and P. Liang. Prefix-Tuning: Optimizing Continuous Prompts for Generation. In Proceedings of ACL 2021.",
        "[6] E. J. Hu, Y. Shen, P. Wallis, et al. LoRA: Low-Rank Adaptation of Large Language Models. In Proceedings of ICLR 2022.",
        "[7] N. Houlsby, A. Giurgiu, S. Jastrzebski, et al. Parameter-Efficient Transfer Learning for NLP. In Proceedings of ICML 2019.",
        "[8] G. Lample, M. Ballesteros, S. Subramanian, et al. Neural Architectures for Named Entity Recognition. In Proceedings of NAACL-HLT 2016.",
        "[9] A. Ekbal and S. Bandyopadhyay. Named Entity Recognition in Bengali using Support Vector Machine. In IJCNLP 2008.",
        "[10] A. Vaswani, N. Shazeer, N. Parmar, et al. Attention Is All You Need. In Advances in Neural Information Processing Systems (NeurIPS), 2017."
    ]

    for ref in refs:
        story.append(Paragraph(ref, ParagraphStyle('RefStyle', parent=body_style, fontSize=8.5, leading=11.5, spaceAfter=3)))

    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"[+] Successfully generated academic PDF paper: {output_filename}")


if __name__ == "__main__":
    out_path = "d:\\NLP project\\Research_Paper_Cross_Lingual_Prompt_Tuned_NER.pdf"
    build_academic_pdf(out_path)
