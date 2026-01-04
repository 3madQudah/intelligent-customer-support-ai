import torch
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    AutoModelForCausalLM
)

# ================= PATHS =================
BERT_MODEL_NAME = "bert-base-uncased"
GPT_MODEL_NAME = "gpt2"

DEVICE = "cpu"

LABEL_MAP = {
    0: "Negative",
    1: "Neutral",
    2: "Positive"
}
# ========================================


class CustomerSupportPipeline:
    def __init__(self):
        # -------- BERT --------
        self.bert_tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        self.bert_model = AutoModelForSequenceClassification.from_pretrained("bert-base-uncased").to(DEVICE)
        self.bert_model.eval()

        # -------- GPT --------
        self.gpt_tokenizer = AutoTokenizer.from_pretrained("gpt2")
        self.gpt_model = AutoModelForCausalLM.from_pretrained("gpt2").to(DEVICE)
        self.gpt_model.eval()

        self.gpt_tokenizer.pad_token = self.gpt_tokenizer.eos_token

    # ---------- Step 1: Classification ----------
    def classify_review(self, text: str) -> str:
        inputs = self.bert_tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True
        ).to(DEVICE)

        with torch.no_grad():
            outputs = self.bert_model(**inputs)

        label_id = outputs.logits.argmax(dim=1).item()
        return LABEL_MAP[label_id]

    # ---------- Step 2: Generation ----------
    def generate_response(self, text: str, label: str) -> str:
        prompt = (
            f"Customer Issue: {text}\n"
            f"Category: {label}\n"
            f"Support Response:"
        )

        inputs = self.gpt_tokenizer(
            prompt,
            return_tensors="pt"
        ).to(DEVICE)

        with torch.no_grad():
            output = self.gpt_model.generate(
                **inputs,
                max_new_tokens=60,
                temperature=0.7,
                top_p=0.9,
                repetition_penalty=1.3,
                no_repeat_ngram_size=3,
                do_sample=True,
                use_cache=False,  # مهم لـ MPS / CPU
                pad_token_id=self.gpt_tokenizer.eos_token_id,
                eos_token_id=self.gpt_tokenizer.eos_token_id
            )

        return self.gpt_tokenizer.decode(
            output[0],
            skip_special_tokens=True
        )

    # ---------- End-to-End ----------
    def run(self, review: str):
        label = self.classify_review(review)
        response = self.generate_response(review, label)

        return label, response


# =====================================================
# ✅ Global pipeline instance (يُستخدم في Streamlit / API)
# =====================================================
_pipeline = CustomerSupportPipeline()


def run_pipeline(review_text: str):
    """
    End-to-End inference function
    Used by Streamlit / FastAPI / CLI
    """
    category, response = _pipeline.run(review_text)
    return category, response


# ================= TEST =================
if __name__ == "__main__":
    review = "I was charged twice for my order"
    category, response = run_pipeline(review)

    print("\n🔍 Review:")
    print(review)

    print("\n🏷 Predicted Category:")
    print(category)

    print("\n🤖 Generated Response:")
    print(response)