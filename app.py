
import html
import torch
import gradio as gr

from unsloth import FastLanguageModel
from unsloth.chat_templates import get_chat_template


# =====================================================
# 1. Model Configuration & Loading
# =====================================================

MODEL_PATH = "mdanialFarokhi/gemma-3-4b-persian-formalizer"

print("Loading model...")

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name=MODEL_PATH,
    max_seq_length=512,
    dtype=torch.bfloat16,
    load_in_4bit=True,
)

tokenizer = get_chat_template(
    tokenizer,
    chat_template="gemma-3"
)

FastLanguageModel.for_inference(model)

print("Model loaded successfully!")


# =====================================================
# 2. Inference Function
# =====================================================

def convert_to_formal(informal_text):

    if not informal_text or not informal_text.strip():
        return """
        <div class="empty-output" dir="rtl">
            لطفاً ابتدا یک متن محاوره‌ای وارد کنید.
        </div>
        """

    messages = [
        {
            "role": "user",
            "content": (
                "متن محاوره‌ای زیر را به متن رسمی تبدیل کن:\n"
                f"{informal_text}"
            )
        }
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt"
    ).to("cuda")

    with torch.no_grad():

        outputs = model.generate(
            input_ids=inputs,
            max_new_tokens=128,
            use_cache=True,
            do_sample=False,
        )

    input_length = inputs.shape[1]

    generated_tokens = outputs[0][input_length:]

    formal_text = tokenizer.decode(
        generated_tokens,
        skip_special_tokens=True
    ).strip()

    # جلوگیری از اجرای HTML احتمالی در خروجی مدل
    safe_text = html.escape(formal_text)

    # حفظ خطوط و پاراگراف‌ها
    return f"""<div class="output-content" dir="auto">{safe_text}</div>"""


# =====================================================
# 3. Premium CSS
# =====================================================

custom_css = """

/* ================================================
   Global
================================================ */

@import url('https://fonts.googleapis.com/css2?family=Vazirmatn:wght@300;400;500;600;700;800&display=swap');

:root {
    --bg-main: #080d1a;
    --bg-card: rgba(17, 25, 45, 0.88);

    --accent: #38bdf8;
    --accent-purple: #a78bfa;

    --text-main: #f1f5f9;
    --text-muted: #94a3b8;

    --border: rgba(148, 163, 184, 0.15);
}

* {
    box-sizing: border-box;
}

body {
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;

    background:
        radial-gradient(
            circle at 15% 20%,
            rgba(56, 189, 248, 0.12),
            transparent 35%
        ),
        radial-gradient(
            circle at 85% 80%,
            rgba(139, 92, 246, 0.13),
            transparent 35%
        ),
        var(--bg-main) !important;

    color: var(--text-main) !important;

    margin: 0;
    min-height: 100vh;

    overflow-x: hidden;
    overflow-y: auto;
}

/* حذف هدر و فوتر پیش‌فرض Gradio */

footer,
.gradio-container > .main > .wrap > .footer {
    display: none !important;
}

/* ================================================
   Main Application
================================================ */

.gradio-container {
    background: transparent !important;

    max-width: 100% !important;
    min-height: 100vh;

    padding: 12px !important;
}

.main-container {
    width: 100%;
    max-width: 1050px;

    margin: auto !important;

    background: var(--bg-card) !important;

    border: 1px solid var(--border) !important;
    border-radius: 24px !important;

    padding: 24px 30px !important;

    backdrop-filter: blur(20px);

    box-shadow:
        0 20px 60px rgba(0, 0, 0, 0.35),
        0 0 35px rgba(56, 189, 248, 0.04);

    position: relative;
    overflow: hidden;
}

/* خط رنگی بالای کارت */

.main-container::before {
    content: "";

    position: absolute;

    top: 0;
    left: 0;
    right: 0;

    height: 3px;

    background: linear-gradient(
        90deg,
        #38bdf8,
        #818cf8,
        #c084fc
    );
}

/* ================================================
   Header
================================================ */

.header-title {
    text-align: center;

    margin-bottom: 22px;
}

.header-title h1 {
    font-family: 'Vazirmatn', sans-serif !important;

    font-size: clamp(1.6rem, 3vw, 2.3rem);

    font-weight: 800;

    margin: 0 0 8px 0;

    background: linear-gradient(
        90deg,
        #38bdf8,
        #818cf8,
        #c084fc
    );

    -webkit-background-clip: text;
    background-clip: text;

    -webkit-text-fill-color: transparent;
}

.header-title p {
    color: #cbd5e1;

    font-size: 0.95rem;
    line-height: 1.8;

    margin: 0;
}

/* ================================================
   Metrics
================================================ */

.metrics-container {
    display: grid;

    grid-template-columns: repeat(5, minmax(0, 1fr));

    gap: 10px;

    margin-bottom: 24px;

    direction: ltr;
}

.metric-card {
    background: rgba(15, 23, 42, 0.72);

    border: 1px solid rgba(56, 189, 248, 0.15);

    border-radius: 14px;

    padding: 12px 8px;

    text-align: center;

    transition: all 0.25s ease;

    min-width: 0;
}

.metric-card:hover {
    transform: translateY(-3px);

    border-color: rgba(56, 189, 248, 0.45);

    box-shadow:
        0 8px 20px rgba(56, 189, 248, 0.08);
}

.metric-value {
    font-family: 'Vazirmatn', sans-serif;

    font-size: clamp(1rem, 2vw, 1.35rem);

    font-weight: 800;

    color: #38bdf8;

    direction: ltr;
}

.metric-label {
    font-family: 'Vazirmatn', sans-serif;

    font-size: 0.75rem;

    color: #94a3b8;

    margin-top: 4px;

    direction: ltr;
}

/* ================================================
   Input / Output Layout
================================================ */

.content-row {
    gap: 18px !important;

    align-items: stretch !important;
}

.content-panel {
    background: rgba(15, 23, 42, 0.45);

    border: 1px solid rgba(148, 163, 184, 0.12);

    border-radius: 16px;

    padding: 16px !important;

    min-width: 0;
}

/* ================================================
   Labels
================================================ */

.rtl-label label {
    font-family: 'Vazirmatn', sans-serif !important;

    color: #c4b5fd !important;

    font-weight: 700 !important;

    font-size: 0.95rem !important;

    text-align: right !important;

    direction: rtl !important;

    margin-bottom: 8px !important;
}

/* ================================================
   Textboxes
================================================ */

.rtl-textbox textarea {
    font-family: 'Vazirmatn', Tahoma, sans-serif !important;

    font-size: 0.95rem !important;

    line-height: 2 !important;

    color: #f1f5f9 !important;

    background: rgba(8, 15, 30, 0.8) !important;

    border: 1px solid #334155 !important;

    border-radius: 12px !important;

    padding: 14px !important;

    direction: rtl !important;

    text-align: right !important;

    /* مدیریت بهتر متن‌های ترکیبی */

    unicode-bidi: plaintext !important;

    transition: all 0.25s ease !important;

    resize: vertical !important;
}

.rtl-textbox textarea:focus {
    border-color: #38bdf8 !important;

    box-shadow:
        0 0 0 3px rgba(56, 189, 248, 0.1) !important;
}

/* ================================================
   Output HTML
================================================ */

.output-box {
    background: rgba(8, 15, 30, 0.8);

    border: 1px solid #334155;

    border-radius: 12px;

    min-height: 155px;

    color: #f1f5f9;

    overflow-wrap: anywhere;
}

.output-content {
    font-family: 'Vazirmatn', Tahoma, sans-serif;

    font-size: 0.95rem;

    line-height: 2.1;

    color: #e2e8f0;

    white-space: pre-wrap;

    text-align: start;

    direction: auto;

    unicode-bidi: plaintext;

    overflow-wrap: anywhere;

    word-break: normal;
}

.empty-output {
    font-family: 'Vazirmatn', sans-serif;

    color: #64748b;

    text-align: right;

    direction: rtl;

    line-height: 2;
}

/* ================================================
   Primary Button
================================================ */

button.primary-btn {
    font-family: 'Vazirmatn', sans-serif !important;

    background: linear-gradient(
        100deg,
        #0284c7,
        #6366f1,
        #8b5cf6
    ) !important;

    color: white !important;

    border: none !important;

    border-radius: 12px !important;

    padding: 12px 20px !important;

    font-size: 0.95rem !important;

    font-weight: 700 !important;

    width: 100% !important;

    min-height: 48px;

    box-shadow:
        0 6px 20px rgba(99, 102, 241, 0.2);

    transition: all 0.25s ease !important;
}

button.primary-btn:hover {
    transform: translateY(-2px);

    box-shadow:
        0 10px 25px rgba(99, 102, 241, 0.35) !important;

    filter: brightness(1.1);
}

button.primary-btn:active {
    transform: scale(0.98);
}

/* ================================================
   Examples
================================================ */

.examples-title {
    font-family: 'Vazirmatn', sans-serif;

    font-size: 0.85rem;

    font-weight: 700;

    color: #94a3b8;

    text-align: right;

    direction: rtl;
}

.examples-rtl {
    direction: rtl !important;

    text-align: right !important;
}

.examples-rtl button {
    font-family: 'Vazirmatn', sans-serif !important;

    background: rgba(30, 41, 59, 0.65) !important;

    border: 1px solid rgba(148, 163, 184, 0.18) !important;

    border-radius: 10px !important;

    color: #cbd5e1 !important;

    font-size: 0.8rem !important;

    text-align: right !important;

    transition: all 0.2s ease !important;
}

.examples-rtl button:hover {
    background: rgba(56, 189, 248, 0.08) !important;

    border-color: #38bdf8 !important;

    color: #f1f5f9 !important;
}

/* ================================================
   Scrollbar
================================================ */

::-webkit-scrollbar {
    width: 7px;
}

::-webkit-scrollbar-track {
    background: #080d1a;
}

::-webkit-scrollbar-thumb {
    background: #334155;

    border-radius: 10px;
}

::-webkit-scrollbar-thumb:hover {
    background: #475569;
}

/* ================================================
   Responsive
================================================ */

@media (max-width: 900px) {

    .main-container {
        max-width: 750px;

        padding: 20px !important;
    }

    .metrics-container {
        grid-template-columns: repeat(3, minmax(0, 1fr));
    }

    .content-row {
        flex-direction: column !important;
    }
}

@media (max-width: 600px) {

    .gradio-container {
        padding: 8px !important;
    }

    .main-container {
        padding: 16px !important;

        border-radius: 18px !important;
    }

    .header-title h1 {
        font-size: 1.5rem;
    }

    .header-title p {
        font-size: 0.8rem;
    }

    .metrics-container {
        grid-template-columns: repeat(2, minmax(0, 1fr));

        gap: 8px;
    }

    .metric-card {
        padding: 10px 5px;
    }

    .content-panel {
        padding: 12px !important;
    }
}

/* لپ‌تاپ‌هایی با ارتفاع کم */

@media (max-height: 800px) and (min-width: 901px) {

    .main-container {
        padding: 16px 24px !important;
    }

    .header-title {
        margin-bottom: 14px;
    }

    .metrics-container {
        margin-bottom: 16px;
    }

    .content-panel {
        padding: 12px !important;
    }

    .rtl-textbox textarea {
        min-height: 120px !important;
    }
}

/* ================================================
   Correct Input / Output Direction
================================================ */

/* ورودی سمت راست و خروجی سمت چپ */

.content-row {
    direction: rtl !important;
    display: flex !important;
    flex-direction: row !important;
    align-items: stretch !important;
    gap: 18px !important;
}

/* حفظ ترتیب ستون‌ها */

#input-panel {
    order: 1 !important;
}

#output-panel {
    order: 2 !important;
}

/* ================================================
   Examples RTL Alignment
================================================ */

.examples-rtl {
    direction: rtl !important;
    text-align: right !important;
}

/* چینش اصلی مثال‌ها */

.examples-rtl > div {
    direction: rtl !important;
    text-align: right !important;
}

/* ردیف مثال‌ها */

.examples-rtl .grid {
    direction: rtl !important;
}

/* هر مثال */

.examples-rtl button {
    direction: rtl !important;
    text-align: right !important;
    justify-content: flex-start !important;
}

/* ================================================
   Responsive
================================================ */

@media (max-width: 900px) {

    .content-row {
        flex-direction: column !important;
    }

    #input-panel,
    #output-panel {
        order: unset !important;
    }
}

/* ================================================
   Remove Top White Space
================================================ */

/* حذف فاصله پیش‌فرض مرورگر */

html,
body {
    margin: 0 !important;
    padding: 0 !important;
}

/* حذف فاصله بالای کانتینر Gradio */

.gradio-container {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* حذف فاصله از ساختار اصلی Gradio */

.gradio-container > .main,
.gradio-container > .main > .wrap {
    padding-top: 0 !important;
    margin-top: 0 !important;
}

/* شروع کارت از بالاترین نقطه */

.main-container {
    margin-top: 0 !important;
}

/* حذف فاصله احتمالی ردیف‌ها و ستون‌های اولیه */

.gradio-container .main > div {
    margin-top: 0 !important;
}


"""


# =====================================================
# 4. Build Gradio Interface
# =====================================================

with gr.Blocks(
    css=custom_css,
    theme=gr.themes.Base(
        primary_hue="sky",
        secondary_hue="violet",
        neutral_hue="slate"
    ),
    title="Gemma Persian Formalizer"
) as demo:

    with gr.Column(elem_classes="main-container"):

        # ---------------------------------------------
        # Header
        # ---------------------------------------------

        gr.HTML("""
        <div class="header-title">

            <h1>Gemma 3 Persian Formalizer</h1>

            <p dir="rtl">
                تبدیل هوشمند متن محاوره‌ای به زبان رسمی
                با استفاده از مدل Gemma 3 4B
            </p>

        </div>
        """)

        # ---------------------------------------------
        # Metrics
        # ---------------------------------------------

        gr.HTML("""
        <div class="metrics-container">

            <div class="metric-card">
                <div class="metric-value">0.96</div>
                <div class="metric-label">BERTScore</div>
            </div>

            <div class="metric-card">
                <div class="metric-value">88.79</div>
                <div class="metric-label">chrF</div>
            </div>

            <div class="metric-card">
                <div class="metric-value">87.92</div>
                <div class="metric-label">METEOR</div>
            </div>

            <div class="metric-card">
                <div class="metric-value">72.99</div>
                <div class="metric-label">BLEU</div>
            </div>

            <div class="metric-card">
                <div class="metric-value">0.248</div>
                <div class="metric-label">Eval Loss</div>
            </div>

        </div>
        """)

        # ---------------------------------------------
        # Input / Output
        # ---------------------------------------------

        with gr.Row(
            elem_classes="content-row",
            equal_height=True
        ):

            # Input panel - RIGHT SIDE

            with gr.Column(
                scale=1,
                elem_classes="content-panel",
                elem_id="input-panel"
            ):

                input_text = gr.Textbox(
                    lines=5,
                    max_lines=8,
                    label="✍️ متن محاوره‌ای",
                    placeholder="متن محاوره‌ای خود را اینجا وارد کنید...",
                    elem_classes=["rtl-textbox", "rtl-label"],
                    elem_id="input-text"
                )

                submit_btn = gr.Button(
                    "✨ تبدیل به متن رسمی",
                    elem_classes="primary-btn"
                )

            # Output panel - LEFT SIDE

            with gr.Column(
                scale=1,
                elem_classes="content-panel",
                elem_id="output-panel"
            ):

                gr.HTML("""
                <div class="examples-title">
                    📄 متن رسمی
                </div>
                """)

                output_text = gr.HTML(
                    value="""<div class="output-content" dir="auto"> متن رسمی تولیدشده در این بخش نمایش داده می‌شود...</div>""",
                    elem_classes=["output-box"],
                    elem_id="output-text"
                )


                # ---------------------------------------------
                # Examples
                # ---------------------------------------------

                gr.HTML("""
                <div class="examples-title">
                    💡 برای شروع، یکی از مثال‌های زیر را انتخاب کنید
                </div>
                """)

                with gr.Column(elem_classes=["examples-rtl"]):
                    gr.Examples(
                        examples=[
                            ["خدا شاهده که من چقدر برای این کار تلاش کردم"],
                            ["اینترنت مون تو یه هفته تموم شد."],
                            ["اگه کارمندا تو خونه صبحونه بخورن بازدهیشون خیلی بیشتره"]
                        ],
                        inputs=input_text,
                
                    )

    # ---------------------------------------------
    # Button Event
    # ---------------------------------------------

    submit_btn.click(
        fn=convert_to_formal,
        inputs=input_text,
        outputs=output_text
    )


# =====================================================
# 5. Launch
# =====================================================

if __name__ == "__main__":
    demo.launch(server_name="0.0.0.0", server_port=7860)