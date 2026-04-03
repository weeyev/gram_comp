import gradio as gr
import torch
from datasets import load_dataset
from models import MLP as Model

model = Model()


def run_inference(sample_idx=0):
    # Use streaming mode to avoid downloading 14GB
    dataset = load_dataset(
        "gram-competition/warped-ifw", streaming=True, trust_remote_code=True
    )

    for i, sample in enumerate(dataset["train"]):
        if i == sample_idx:
            break

    return f"Sample {sample_idx} processed. Keys: {list(sample.keys())}"


demo = gr.Interface(
    fn=run_inference,
    inputs=gr.Slider(0, 100, step=1, label="Sample Index"),
    outputs="text",
    title="GRaM Competition - Warped IFW",
)

demo.launch()
