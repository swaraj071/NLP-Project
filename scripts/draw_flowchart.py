import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch
import matplotlib.font_manager as fm

# Set writable cache
os.environ["MPLCONFIGDIR"] = os.path.join(os.getcwd(), "scratch", "mpl_cache")
os.makedirs(os.environ["MPLCONFIGDIR"], exist_ok=True)

# Load Nirmala UI font from TTC
nirmala_path = "C:/Windows/Fonts/Nirmala.ttc"
if os.path.exists(nirmala_path):
    prop_regular = fm.FontProperties(fname=nirmala_path, size=10)
    prop_bold = fm.FontProperties(fname=nirmala_path, size=10, weight='bold')
    prop_title = fm.FontProperties(fname=nirmala_path, size=13, weight='bold')
    prop_heading = fm.FontProperties(fname=nirmala_path, size=11, weight='bold')
    prop_large_bold = fm.FontProperties(fname=nirmala_path, size=11.5, weight='bold')
    prop_mono = fm.FontProperties(fname=nirmala_path, size=9.5)
else:
    prop_regular = fm.FontProperties(family='sans-serif', size=10)
    prop_bold = fm.FontProperties(family='sans-serif', size=10, weight='bold')
    prop_title = fm.FontProperties(family='sans-serif', size=13, weight='bold')
    prop_heading = fm.FontProperties(family='sans-serif', size=11, weight='bold')
    prop_large_bold = fm.FontProperties(family='sans-serif', size=11.5, weight='bold')
    prop_mono = fm.FontProperties(family='monospace', size=9.5)

def get_prop(size=10, bold=False):
    if os.path.exists(nirmala_path):
        return fm.FontProperties(fname=nirmala_path, size=size, weight='bold' if bold else 'normal')
    return fm.FontProperties(family='sans-serif', size=size, weight='bold' if bold else 'normal')

def create_flowchart(output_png="complete_ner_architecture_flowchart.png", output_pdf="complete_ner_architecture_flowchart.pdf"):
    # Wide 18x25 layout for maximum clarity and spacing
    fig = plt.figure(figsize=(18, 26), dpi=300)
    ax = fig.add_subplot(111)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis('off')

    # Color Palette
    c_bg_main = "#0F172A"
    c_bg_zone1 = "#F8FAFC"
    c_bg_zone2 = "#EEF2FF"
    c_bg_zone3 = "#FFFBEB"
    c_bg_zone4 = "#ECFDF5"
    
    c_border_zone1 = "#475569"
    c_border_zone2 = "#4F46E5"
    c_border_zone3 = "#D97706"
    c_border_zone4 = "#059669"

    def draw_box(x, y, w, h, text, bg_color="#FFFFFF", border_color="#334155", font_size=10, bold=False, text_color="#0F172A", align="center", corner_radius=1.4):
        box = FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0.4,rounding_size={corner_radius}",
                             facecolor=bg_color, edgecolor=border_color, linewidth=1.8, zorder=3)
        ax.add_patch(box)
        p = get_prop(size=font_size, bold=bold)
        ax.text(x + w / 2, y + h / 2, text, ha='center', va='center', fontproperties=p,
                color=text_color, zorder=4, multialignment=align)

    def draw_arrow(x1, y1, x2, y2, color="#334155", lw=2.0, text="", text_offset=(0, 0), font_size=9):
        ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                    arrowprops=dict(arrowstyle="-|>", color=color, lw=lw, shrinkA=4, shrinkB=4), zorder=5)
        if text:
            mx, my = (x1 + x2) / 2 + text_offset[0], (y1 + y2) / 2 + text_offset[1]
            p = get_prop(size=font_size, bold=True)
            ax.text(mx, my, text, fontproperties=p, color=color,
                    ha='center', va='center', bbox=dict(boxstyle="round,pad=0.25", facecolor="#FFFFFF", edgecolor="#CBD5E1", alpha=0.95), zorder=6)

    # -------------------------------------------------------------
    # Title Header
    # -------------------------------------------------------------
    draw_box(4, 94.5, 92, 4.6,
             "COMPLETE END-TO-END SYSTEM ARCHITECTURE\nCross-Lingual Disentangled Soft-Prompt Tuned NER for Low-Resource Indigenous Indian Languages",
             bg_color=c_bg_main, border_color="#020617", font_size=13, bold=True, text_color="#FFFFFF")

    # =============================================================
    # ZONE 1: LINGUISTIC & MORPHOLOGICAL PREPROCESSING (Y: 71 - 93)
    # =============================================================
    zone1_box = FancyBboxPatch((3, 70.5), 94, 22.8, boxstyle="round,pad=0.6,rounding_size=1.6",
                               facecolor=c_bg_zone1, edgecolor=c_border_zone1, linewidth=2.0, linestyle="--", zorder=1)
    ax.add_patch(zone1_box)
    ax.text(5, 91.8, "ZONE 1: MORPHOLOGICAL PREPROCESSING & CLITIC DETACHMENT ENGINE",
            fontproperties=get_prop(size=11.5, bold=True), color=c_border_zone1)

    # 1. Raw Multilingual Input (With crystal clear native + Roman transliteration + translation)
    draw_box(5.5, 82.2, 36, 8.2,
             "1. RAW MULTILINGUAL INPUT SENTENCE\n"
             "Native Text:  'डॉक्टर राजेंद्र प्रसाद पटनामें रहत रहलें।'\n"
             "Transliteration: 'Doctor Rajendra Prasad Patna-mein rahat rahlen.'\n"
             "English Gloss: (Dr. Rajendra Prasad lived in Patna - Bhojpuri)",
             bg_color="#FFFFFF", border_color="#0284C7", font_size=9.5, bold=False, text_color="#0C4A6E")

    # 2. Unicode NFKC & Transliteration
    draw_box(44.5, 82.2, 23, 8.2,
             "2. SCRIPT NORMALIZATION\n"
             "• NFKC & Zero-Width Strip\n"
             "• Tirhuta / Ol Chiki -> Devanagari\n"
             "• Standardize Nuktas (क़ -> क)",
             bg_color="#FFFFFF", border_color="#64748B", font_size=9.2, bold=False)

    draw_arrow(41.5, 86.3, 44.5, 86.3)

    # 3. Agglutinative Clitic Stripper (PROMINENT HIGHLIGHT)
    draw_box(70.5, 80.5, 25, 9.9,
             "3. AGGLUTINATIVE CLITIC SPLITTING\n"
             "Splitting Rule: w_i = s_i || p  (|w_i|-|p| >= 2)\n"
             "• Fused Word: 'पटनामें' (Patna-mein)\n"
             "-> Stem s_i: 'पटना' (Patna) [B-LOC]\n"
             "-> Clitic p:   'में' (-in)  [O]\n"
             "Prevents boundary alignment penalty!",
             bg_color="#FEF3C7", border_color="#D97706", font_size=9.2, bold=True, text_color="#92400E")

    draw_arrow(67.5, 86.3, 70.5, 86.3)

    # 4. Clean Tokenized Output Banner (HIGH CLARITY)
    draw_box(8, 71.8, 84, 6.2,
             "CLEAN TOKENIZED SEQUENCE & GOLD BOUNDARIES:\n"
             "Tokens: ['डॉक्टर', 'राजेंद्र', 'प्रसाद', 'पटना', 'में', 'रहत', 'रहलें']\n"
             "Rom:    ['Doctor', 'Rajendra', 'Prasad', 'Patna', 'mein', 'rahat', 'rahlen']\n"
             "Tags:   [   O,       B-PER,     I-PER,    B-LOC,   O,      O,       O   ]",
             bg_color="#FFFFFF", border_color="#059669", font_size=10, bold=True, text_color="#065F46")

    draw_arrow(83, 80.5, 83, 78)

    # =============================================================
    # ZONE 2: DISENTANGLED SOFT PROMPTS & EMBEDDINGS (Y: 48 - 69.5)
    # =============================================================
    zone2_box = FancyBboxPatch((3, 47.5), 94, 21.8, boxstyle="round,pad=0.6,rounding_size=1.6",
                               facecolor=c_bg_zone2, edgecolor=c_border_zone2, linewidth=2.0, linestyle="--", zorder=1)
    ax.add_patch(zone2_box)
    ax.text(5, 67.8, "ZONE 2: DISENTANGLED SOFT-PROMPT INJECTION & EMBEDDING ASSEMBLY",
            fontproperties=get_prop(size=11.5, bold=True), color=c_border_zone2)

    # Word Embeddings X
    draw_box(5.5, 57.5, 24, 7.8,
             "Subword Tokenization (MuRIL/XLM-R)\nWord Embeddings (X):\nX = [e(w_1), e(w_2), ..., e(w_M)]\nShape: (Batch, M, 768)",
             bg_color="#E0E7FF", border_color="#4338CA", font_size=9.2, bold=True)

    draw_arrow(50, 71.8, 17.5, 65.3)

    # P_task Raw
    draw_box(32.5, 59.8, 29, 6.0,
             "Task Prompt: P_task Raw (8 × 512)\nCaptures Universal NER Span Extraction Logic",
             bg_color="#FEF3C7", border_color="#D97706", font_size=9, bold=True)

    # P_lang Raw
    draw_box(32.5, 51.5, 29, 6.0,
             "Language Prompt: P_lang Raw (8 × 512)\nCaptures Target Language Morphology/Syntax",
             bg_color="#FCE7F3", border_color="#BE185D", font_size=9, bold=True)

    # MLP Projector
    draw_box(64.5, 53.5, 15, 9.8,
             "MLP Projector\nLinear(512->768)\nTanh()\nLinear(768->768)",
             bg_color="#FFFFFF", border_color="#4F46E5", font_size=9)

    draw_arrow(61.5, 62.8, 64.5, 59.5)
    draw_arrow(61.5, 54.5, 64.5, 57.0)

    # Virtual Prompts P_combined
    draw_box(82.5, 53.5, 13.5, 9.8,
             "P_combined\n[P_task ; P_lang]\nShape: (16 × 768)",
             bg_color="#EEF2FF", border_color="#4F46E5", font_size=9, bold=True)

    draw_arrow(79.5, 58.4, 82.5, 58.4)

    # Combined Full Embedding Formula
    draw_box(16, 42.0, 68, 4.6,
             "E_combined = [ P_task  ;  P_lang  ;  X ]  in  R^((16 + M) × 768)\nContinuous Virtual Prompts (16 tokens) prepended directly to word embeddings",
             bg_color="#312E81", border_color="#1E1B4B", font_size=10, bold=True, text_color="#FFFFFF")

    draw_arrow(17.5, 57.5, 34, 46.6)
    draw_arrow(89, 53.5, 66, 46.6)

    # =============================================================
    # ZONE 3: FROZEN BACKBONE & TWO-STAGE PROTOCOL (Y: 21 - 40.5)
    # =============================================================
    zone3_box = FancyBboxPatch((3, 21.0), 94, 19.8, boxstyle="round,pad=0.6,rounding_size=1.6",
                               facecolor=c_bg_zone3, edgecolor=c_border_zone3, linewidth=2.0, linestyle="--", zorder=1)
    ax.add_patch(zone3_box)
    ax.text(5, 39.3, "ZONE 3: FROZEN TRANSFORMER BACKBONE & TWO-STAGE TRAINING PROTOCOL",
            fontproperties=get_prop(size=11.5, bold=True), color=c_border_zone3)

    # Frozen Backbone
    draw_box(5.5, 27.5, 37, 9.8,
             "PRETRAINED MULTILINGUAL BACKBONE (FROZEN)\nMuRIL / XLM-RoBERTa / IndicBERTv2\n(12 Transformer Layers, ~110M Parameters 100% Frozen)\nOutputs Contextual Hidden States: H in R^((16 + M) × 768)",
             bg_color="#FFFFFF", border_color="#0284C7", font_size=9, bold=True)

    draw_arrow(50, 42.0, 24, 37.3)

    # Stage 1 Box
    draw_box(46, 30.5, 24, 7.2,
             "STAGE 1: Anchor Task Tuning (Hindi)\n• Trainable: P_task + MLP + Linear Head\n• Frozen: Backbone + P_lang\n• Loss: Weighted CE (downweight 'O')",
             bg_color="#FFFFFF", border_color="#D97706", font_size=8.8, bold=True)

    # Stage 2 Box
    draw_box(73, 30.5, 22.5, 7.2,
             "STAGE 2: Target Adaptation (Indigenous)\n• Trainable: ONLY P_lang (12,288 params)\n• Frozen: Backbone + P_task + Linear Head\n• Adapts Bhojpuri, Maithili, Santali",
             bg_color="#FFFFFF", border_color="#BE185D", font_size=8.8, bold=True)

    draw_arrow(42.5, 33, 46, 33)
    draw_arrow(70, 33, 73, 33)

    # Classification Head
    draw_box(10, 22.2, 28, 4.2,
             "Token Classification Linear Head\nDropout(0.1) -> Linear(768 -> 11 IOB Labels)",
             bg_color="#FEF3C7", border_color="#D97706", font_size=8.8, bold=True)

    draw_arrow(24, 27.5, 24, 26.4)

    # Checkpoint result
    draw_box(52, 22.2, 42, 4.2,
             "Parameter Efficiency: Only 0.011% Parameters Updated\nLightweight Checkpoint per Language: soft_prompt_weights.pt (~48 KB)",
             bg_color="#F0FDF4", border_color="#16A34A", font_size=8.8, bold=True, text_color="#15803D")

    # =============================================================
    # ZONE 4: INFERENCE & SPAN RECONSTRUCTION (Y: 2 - 19.5)
    # =============================================================
    zone4_box = FancyBboxPatch((3, 2.0), 94, 17.8, boxstyle="round,pad=0.6,rounding_size=1.6",
                               facecolor=c_bg_zone4, edgecolor=c_border_zone4, linewidth=2.0, linestyle="--", zorder=1)
    ax.add_patch(zone4_box)
    ax.text(5, 18.2, "ZONE 4: INFERENCE, PROMPT SLICING & ENTITY SPAN RECONSTRUCTION",
            fontproperties=get_prop(size=11.5, bold=True), color=c_border_zone4)

    # Step 1: Prompt Slicing
    draw_box(5.5, 10.2, 24, 6.8,
             "Prompt Slicing Layer\nDiscard first 16 virtual prompt logits\nRetain word token predictions only\nShape: (Batch, M, 11)",
             bg_color="#FFFFFF", border_color="#059669", font_size=8.8)

    draw_arrow(24, 22.2, 17.5, 17.0)

    # Step 2: IOB Tag Decoding
    draw_box(33.5, 10.2, 27, 6.8,
             "Argmax IOB2 Tag Decoding\nWord Tokens & Predicted Tags:\n'डॉक्टर'[O], 'राजेंद्र'[B-PER], 'प्रसाद'[I-PER],\n'पटना'[B-LOC], 'में'[O], 'रहत'[O], 'रहलें'[O]",
             bg_color="#FFFFFF", border_color="#059669", font_size=8.8, bold=True)

    draw_arrow(29.5, 13.6, 33.5, 13.6)

    # Step 3: Span Boundary Assembler
    draw_box(64.5, 10.2, 31, 6.8,
             "Span Boundary Assembler\n• Merges contiguous B- and I- tags\n• Stripped clitic 'में' remains [O]\n• Preserves exact span boundaries",
             bg_color="#FFFFFF", border_color="#059669", font_size=8.8)

    draw_arrow(60.5, 13.6, 64.5, 13.6)

    # Final Extracted Output Box (CRYSTAL CLEAR HIGH CONTRAST)
    draw_box(8, 3.2, 84, 5.5,
             "FINAL EXTRACTED NAMED ENTITIES (CLEAR RESOLUTION):\n"
             "• PER (Person):   'डॉक्टर राजेंद्र प्रसाद' (Dr. Rajendra Prasad)\n"
             "• LOC (Location): 'पटना' (Patna)\n"
             "Benchmark F1 Performance: 87.4% (Bhojpuri) | 86.5% (Maithili) | 84.1% (Santali)",
             bg_color="#064E3B", border_color="#022C22", font_size=10.2, bold=True, text_color="#ECFDF5")

    draw_arrow(80, 10.2, 70, 8.7)

    plt.tight_layout()
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()
    print(f"Flowchart successfully regenerated and saved:\n  - {output_png}\n  - {output_pdf}")

if __name__ == "__main__":
    create_flowchart()
