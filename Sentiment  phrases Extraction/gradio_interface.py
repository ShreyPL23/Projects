import gradio as gr
import torch
from transformers import AutoModelForQuestionAnswering, AutoTokenizer

# Load the fine-tuned model and tokenizer
# model_path = "./finetune-BERT-tweet/checkpoint-375"
# model_path = "./bert_model"
model_path = "finetuned_bert_model-fulldataset"

model = AutoModelForQuestionAnswering.from_pretrained(model_path)
tokenizer = AutoTokenizer.from_pretrained(model_path)

# Define function for prediction
def predict_sentiment(tweet, sentiment):
    inputs = tokenizer(
        sentiment,  # the sentiment question
        tweet,      # the tweet context
        padding=True,
        truncation=True,
        return_tensors="pt"
    )

    model.eval()
    with torch.no_grad():
        outputs = model(**inputs)
        start_logits = outputs.start_logits
        end_logits = outputs.end_logits

        start_position = torch.argmax(start_logits, dim=-1).item()
        end_position = torch.argmax(end_logits, dim=-1).item()

        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        sentiment_span = tokens[start_position:end_position+1]
        sentiment_text = tokenizer.convert_tokens_to_string(sentiment_span)
    
    return sentiment_text

# Create Gradio interface
chat_application = gr.Interface(
    fn=predict_sentiment,
    inputs=[
        gr.Textbox(label="Enter Tweet"),
        gr.Textbox(label="Enter Sentiment"),
    ],
    outputs=gr.Textbox(label="Predicted Sentiment Span"),
    title="Tweet Sentiment Analysis",
    description="Enter a tweet and the sentiment type (e.g., positive, negative, neutral) to predict the relevant sentiment span."
)

# Launch the app
chat_application.launch(server_name="127.0.0.1", server_port= 7860)
