import os
import sys
from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, KeepTogether, HRFlowable
)
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.pdfgen import canvas

class NumberedCanvas(canvas.Canvas):
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
            self.draw_page_number(num_pages)
            super().showPage()
        super().save()

    def draw_page_number(self, page_count):
        if self._pageNumber == 1:
            return  # Suppress header/footer on cover page
        self.saveState()
        self.setFont("Helvetica", 8)
        self.setFillColor(colors.HexColor("#64748B"))
        
        # Header
        self.drawString(54, letter[1] - 36, "Cross-Lingual Prompt-Tuned LLMs for NER in Indigenous Indian Languages")
        self.setStrokeColor(colors.HexColor("#CBD5E1"))
        self.setLineWidth(0.5)
        self.line(54, letter[1] - 42, letter[0] - 54, letter[1] - 42)
        
        # Footer
        page_text = f"Page {self._pageNumber} of {page_count}"
        self.drawRightString(letter[0] - 54, 36, page_text)
        self.drawString(54, 36, "Comprehensive Technical Architecture & Implementation Guide")
        self.line(54, 48, letter[0] - 54, 48)
        self.restoreState()

def build_pdf(filename="Cross_Lingual_Prompt_Tuned_NER_Guide.pdf"):
    doc = SimpleDocTemplate(
        filename,
        pagesize=letter,
        leftMargin=54,
        rightMargin=54,
        topMargin=54,
        bottomMargin=54
    )
    
    styles = getSampleStyleSheet()
    
    # Custom Palette
    c_primary = colors.HexColor("#1E1B4B")      # Dark Indigo
    c_secondary = colors.HexColor("#4F46E5")    # Indigo Accent
    c_dark = colors.HexColor("#0F172A")         # Slate Dark Body
    c_light_bg = colors.HexColor("#F8FAFC")     # Light slate background
    c_code_bg = colors.HexColor("#1E293B")      # Dark code background
    c_code_fg = colors.HexColor("#38BDF8")      # Light blue code text
    c_border = colors.HexColor("#E2E8F0")       # Border grey
    
    # Typography Styles
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=24,
        leading=28,
        textColor=c_primary,
        spaceAfter=10
    )
    
    subtitle_style = ParagraphStyle(
        'CoverSubtitle',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        spaceAfter=20
    )
    
    h1_style = ParagraphStyle(
        'Heading1_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=15,
        leading=19,
        textColor=c_primary,
        spaceBefore=16,
        spaceAfter=8,
        keepWithNext=True
    )

    h2_style = ParagraphStyle(
        'Heading2_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Bold',
        fontSize=12,
        leading=16,
        textColor=c_secondary,
        spaceBefore=12,
        spaceAfter=6,
        keepWithNext=True
    )
    
    body_style = ParagraphStyle(
        'Body_Custom',
        parent=styles['Normal'],
        fontName='Helvetica',
        fontSize=9.5,
        leading=14,
        textColor=c_dark,
        spaceAfter=8
    )

    bullet_style = ParagraphStyle(
        'Bullet_Custom',
        parent=body_style,
        leftIndent=15,
        bulletIndent=5,
        spaceAfter=4
    )

    code_style = ParagraphStyle(
        'Code_Custom',
        parent=styles['Normal'],
        fontName='Courier',
        fontSize=8,
        leading=11,
        textColor=c_code_fg,
        spaceBefore=4,
        spaceAfter=4
    )

    callout_style = ParagraphStyle(
        'Callout_Custom',
        parent=styles['Normal'],
        fontName='Helvetica-Oblique',
        fontSize=9,
        leading=13,
        textColor=colors.HexColor("#334155"),
        spaceBefore=4,
        spaceAfter=4
    )

    story = []

    # Title Block
    story.append(Paragraph("Cross-Lingual Prompt-Tuned LLMs for NER in Indigenous Indian Languages", title_style))
    story.append(Paragraph("System Architecture, Mathematical Formulations & Component-by-Component Guide", subtitle_style))
    story.append(HRFlowable(width="100%", thickness=2, color=c_secondary, spaceBefore=0, spaceAfter=15))

    # Executive Summary / Overview
    story.append(Paragraph("1. System Overview & Core Philosophy", h1_style))
    story.append(Paragraph(
        "This project implements a modular, production-ready system for <b>Named Entity Recognition (NER)</b> "
        "tailored to low-resource Indigenous Indian languages (including <i>Bhojpuri, Maithili, Santali, Hindi, Marathi, Bengali, Gujarati, Odia, and Punjabi</i>). "
        "Standard fine-tuning of large multilingual models (e.g., MuRIL, XLM-RoBERTa, IndicBERTv2) on small target datasets often causes severe catastrophic forgetting and high storage overhead. "
        "To solve this, the architecture employs <b>Disentangled Soft-Prompt Tuning (P<sub>task</sub> + P<sub>lang</sub>)</b> combined with <b>Script Normalization</b> and <b>Agglutinative Postposition Stripping</b>.",
        body_style
    ))

    # Architecture Box / Table
    arch_summary_data = [
        [Paragraph("<b>Component</b>", body_style), Paragraph("<b>Module Path</b>", body_style), Paragraph("<b>Primary Function</b>", body_style)],
        [Paragraph("Script Normalizer", body_style), Paragraph("<code>data/normalizer.py</code>", body_style), Paragraph("NFKC normalization, zero-width removal, Ol Chiki/Tirhuta transliteration & nukta standardization.", body_style)],
        [Paragraph("Postposition Stripper", body_style), Paragraph("<code>data/normalizer.py</code>", body_style), Paragraph("Detaches agglutinative case clitics (-में, -से, -के) to preserve entity span boundaries.", body_style)],
        [Paragraph("PyTorch Dataset Loader", body_style), Paragraph("<code>data/loader.py</code>", body_style), Paragraph("Subword label alignment, first-subword tagging, and virtual prompt index shifting (-100).", body_style)],
        [Paragraph("Synthetic Generator", body_style), Paragraph("<code>data/synthetic.py</code>", body_style), Paragraph("Multi-lingual template slot filling generating 11-class IOB annotated datasets.", body_style)],
        [Paragraph("Disentangled Soft Prompt", body_style), Paragraph("<code>models/soft_prompt.py</code>", body_style), Paragraph("Prepends learnable P<sub>task</sub> and P<sub>lang</sub> continuous embedding tensors with optional MLP projection.", body_style)],
        [Paragraph("Prompt-Tuned NER Model", body_style), Paragraph("<code>models/ner_model.py</code>", body_style), Paragraph("Wraps transformer backbone with soft prompt injection and linear token classification head.", body_style)],
        [Paragraph("Multi-Stage Trainer", body_style), Paragraph("<code>training/trainer.py</code>", body_style), Paragraph("Stage 1 anchor task tuning (P<sub>task</sub>) followed by Stage 2 language adaptation (P<sub>lang</sub>).", body_style)],
        [Paragraph("Span Evaluator", body_style), Paragraph("<code>evaluation/metrics.py</code>", body_style), Paragraph("Computes span-level seqeval Precision, Recall, and F1 scores while ignoring -100 slots.", body_style)],
        [Paragraph("Streamlit Dashboard", body_style), Paragraph("<code>app.py</code>", body_style), Paragraph("Interactive GUI for real-time text normalizer diagnostic, prompt control, and badge rendering.", body_style)]
    ]
    t_arch = Table(arch_summary_data, colWidths=[1.3*inch, 1.4*inch, 4.3*inch])
    t_arch.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_light_bg),
        ('TEXTCOLOR', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_arch)
    story.append(Spacer(1, 15))

    # Section 2: Data Pre-processing Pipeline
    story.append(Paragraph("2. Deep-Dive: Script Normalization & Postposition Processing", h1_style))
    story.append(Paragraph(
        "Low-resource Indian scripts suffer from two major representation challenges: script divergence (e.g., Santali in Ol Chiki vs. Devanagari) and agglutinative clitic attachment.",
        body_style
    ))
    
    story.append(Paragraph("A. ScriptNormalizer (data/normalizer.py)", h2_style))
    story.append(Paragraph("The <code>ScriptNormalizer</code> applies a 5-step sanitization sequence to input text:", body_style))
    story.append(Paragraph("• <b>NFKC Unicode Normalization:</b> Standardizes visually identical Unicode representations (e.g., combining characters vs. precomposed glyphs).", bullet_style))
    story.append(Paragraph("• <b>Zero-Width Character Cleanup:</b> Strips non-printable formatting artifacts (<code>\\u200b</code>, <code>\\u200c</code>, <code>\\u200d</code>, <code>\\ufeff</code>).", bullet_style))
    story.append(Paragraph("• <b>Cross-Script Transliteration:</b> Maps Ol Chiki (Santali) and Tirhuta (Maithili) character ranges to Devanagari equivalents to enable cross-lingual embedding sharing.", bullet_style))
    story.append(Paragraph("• <b>Nukta Standardization:</b> Unifies dot diacritics (e.g., क़ → क, ख़ → ख, ग़ → ग, ज़ → ज, फ़ → फ) to reduce out-of-vocabulary subword fragmentation.", bullet_style))
    story.append(Paragraph("• <b>Whitespace Collapsing:</b> Replaces multiple whitespace spaces with a single space.", bullet_style))

    story.append(Paragraph("B. AgglutinativePostProcessor (data/normalizer.py)", h2_style))
    story.append(Paragraph(
        "In Indic languages like Bhojpuri, Maithili, and Hindi, case postpositions are frequently written attached directly to noun stems (e.g., <i>'पटनामें'</i> instead of <i>'पटना में'</i>). "
        "If left intact, token classification heads assign the entity tag (e.g., <code>B-LOC</code>) to the entire fused word, corrupting entity span evaluation. "
        "The <code>AgglutinativePostProcessor</code> detaches postposition suffixes (e.g., <i>'में', 'से', 'के', 'को', 'रे', 'ते'</i>) while enforcing a <code>min_stem_len >= 2</code> constraint. "
        "Crucially, it re-aligns token sequences and IOB tags: <i>tokens=['पटना', 'में']</i> with <i>tags=['B-LOC', 'O']</i>.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Section 3: Soft Prompt Architecture & Formulations
    story.append(Paragraph("3. Deep-Dive: Disentangled Soft-Prompt Architecture", h1_style))
    story.append(Paragraph(
        "Instead of fine-tuning all weights of the transformer backbone (which requires updating ~110M+ parameters per target language), "
        "disentangled soft-prompt tuning prepends a small sequence of continuous learnable virtual token embeddings to input word embeddings.",
        body_style
    ))
    story.append(Paragraph("A. Mathematical Formulation", h2_style))
    story.append(Paragraph(
        "Let the input word embeddings extracted from the backbone embedding layer be:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>E<sub>word</sub></b> = [e(w<sub>1</sub>), e(w<sub>2</sub>), ..., e(w<sub>N</sub>)] &nbsp;&in;&nbsp; R<sup>N &times; d<sub>hidden</sub></sup><br/>"
        "We construct two separate prompt parameter matrices:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>P<sub>task</sub></b> &nbsp;&in;&nbsp; R<sup>L<sub>task</sub> &times; d<sub>hidden</sub></sup> &nbsp;&nbsp;&nbsp;&nbsp;(Learns generic NER boundary rules on anchor data)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>P<sub>lang</sub></b> &nbsp;&in;&nbsp; R<sup>L<sub>lang</sub> &times; d<sub>hidden</sub></sup> &nbsp;&nbsp;&nbsp;&nbsp;(Learns target language morphology & script characteristics)<br/>"
        "The concatenated embedding sequence passed to the transformer encoder is:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>E<sub>combined</sub></b> = [ <b>P<sub>task</sub></b> ; <b>P<sub>lang</sub></b> ; <b>E<sub>word</sub></b> ] &nbsp;&in;&nbsp; R<sup>(L<sub>task</sub> + L<sub>lang</sub> + N) &times; d<sub>hidden</sub></sup>",
        body_style
    ))

    story.append(Paragraph("B. MLP Projection Bottleneck (models/soft_prompt.py)", h2_style))
    story.append(Paragraph(
        "Direct optimization of high-dimensional soft prompts (d<sub>hidden</sub>=768) can suffer from optimization instability. "
        "When <code>use_mlp_projection=True</code>, prompts are parameterized via a low-rank bottleneck layer:<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>P'<sub>task</sub></b> &nbsp;&in;&nbsp; R<sup>L<sub>task</sub> &times; d<sub>mid</sub></sup> &nbsp;&nbsp;&nbsp;&nbsp;(where d<sub>mid</sub> = 512)<br/>"
        "&nbsp;&nbsp;&nbsp;&nbsp;<b>P<sub>task</sub></b> = W<sub>2</sub> &middot; Tanh( W<sub>1</sub> &middot; <b>P'<sub>task</sub></b> + b<sub>1</sub> ) + b<sub>2</sub><br/>"
        "This reparameterization drastically speeds up training convergence.",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Section 4: Data Loader & Label Shifting
    story.append(Paragraph("4. Deep-Dive: PyTorch Dataset & Label Index Shifting", h1_style))
    story.append(Paragraph(
        "In <code>data/loader.py</code>, the <code>NERDataset</code> class performs critical subword-to-label alignment and virtual token index shifting:",
        body_style
    ))
    story.append(Paragraph("1. <b>First Subword Alignment:</b> Words are tokenized into subwords. The first subword receives the original token's IOB label ID, while all remaining subwords are assigned label ID <code>-100</code> (ignored in PyTorch CrossEntropyLoss).", bullet_style))
    story.append(Paragraph("2. <b>Prompt Index Shifting:</b> Because <code>(L<sub>task</sub> + L<sub>lang</sub>)</code> virtual prompt embeddings are prepended to the input sequence, the label sequence tensor and attention mask tensor are shifted accordingly:<br/>"
                           "&nbsp;&nbsp;&nbsp;&nbsp;<code>labels = [-100] * prompt_length + [CLS_label] + word_labels + [SEP_label] + padding_labels</code><br/>"
                           "&nbsp;&nbsp;&nbsp;&nbsp;<code>attention_mask = [1] * prompt_length + [1] * seq_len + [0] * pad_len</code>", bullet_style))
    story.append(Spacer(1, 10))

    # Section 5: Multi-Stage Training Protocol
    story.append(Paragraph("5. Deep-Dive: Multi-Stage Training Protocol", h1_style))
    story.append(Paragraph(
        "The <code>MultiStagePromptTrainer</code> (in <code>training/trainer.py</code>) executes a two-stage training scheme that enforces modularity between task knowledge and language knowledge:",
        body_style
    ))
    
    stage_table_data = [
        [Paragraph("<b>Training Stage</b>", body_style), Paragraph("<b>Trainable Parameters</b>", body_style), Paragraph("<b>Frozen Parameters</b>", body_style), Paragraph("<b>Target Objective</b>", body_style)],
        [
            Paragraph("<b>Stage 1: Task Anchor Tuning</b>", body_style),
            Paragraph("P<sub>task</sub> Prompt,<br/>Linear Head", body_style),
            Paragraph("Transformer Backbone,<br/>P<sub>lang</sub> Prompt", body_style),
            Paragraph("Learn entity boundary extraction rules on high-resource anchor language (Hindi/English).", body_style)
        ],
        [
            Paragraph("<b>Stage 2: Language Adaptation</b>", body_style),
            Paragraph("P<sub>lang</sub> Prompt", body_style),
            Paragraph("Transformer Backbone,<br/>P<sub>task</sub> Prompt,<br/>Linear Head", body_style),
            Paragraph("Adapt model syntax & script representations to target low-resource language (Bhojpuri/Maithili/Santali).", body_style)
        ],
        [
            Paragraph("<b>Stage 3: Joint Fine-Tuning (Opt.)</b>", body_style),
            Paragraph("P<sub>task</sub>, P<sub>lang</sub>,<br/>Linear Head", body_style),
            Paragraph("Transformer Backbone", body_style),
            Paragraph("Refine combined task & language representations end-to-end.", body_style)
        ]
    ]
    t_stage = Table(stage_table_data, colWidths=[1.5*inch, 1.4*inch, 1.4*inch, 2.7*inch])
    t_stage.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), c_light_bg),
        ('TEXTCOLOR', (0,0), (-1,0), c_primary),
        ('GRID', (0,0), (-1,-1), 0.5, c_border),
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('TOPPADDING', (0,0), (-1,-1), 5),
        ('BOTTOMPADDING', (0,0), (-1,-1), 5),
    ]))
    story.append(t_stage)
    story.append(Spacer(1, 12))

    # Section 6: Evaluation & Metrics
    story.append(Paragraph("6. Deep-Dive: Span-Level Evaluation Metrics", h1_style))
    story.append(Paragraph(
        "In <code>evaluation/metrics.py</code>, the <code>NEREvaluator</code> decodes predicted logit distributions back to string IOB tags. "
        "During decoding, all index positions with label <code>-100</code> (virtual prompt slots, subword continuations, padding) are filtered out. "
        "The evaluator then calculates <b>exact entity span Precision, Recall, and F1 scores</b> using <code>seqeval</code> (or a robust fallback token-accuracy evaluator if seqeval is uninstalled).",
        body_style
    ))
    story.append(Spacer(1, 10))

    # Section 7: Interactive Dashboard & Execution
    story.append(Paragraph("7. Interactive Dashboard & Verification", h1_style))
    story.append(Paragraph(
        "The system includes a Streamlit GUI (<code>app.py</code>) allowing users to interactively test script normalization, postposition splitting, and prompt configuration.<br/>"
        "• <b>Launch Command:</b> <code>streamlit run app.py</code><br/>"
        "• <b>Sanity Suite Command:</b> <code>python tests/sanity_check.py</code>",
        body_style
    ))

    # Build Document
    doc.build(story, canvasmaker=NumberedCanvas)
    print(f"Successfully generated PDF guide: {filename}")

if __name__ == "__main__":
    build_pdf()
