import gradio as gr
from bert_utils import BERTClassifier
from evaluate import emotion_predict

def emotion_classification(input_data):
    result = emotion_predict(input_data)
    return result

demo = gr.Interface(fn=emotion_classification, inputs=gr.Textbox(),outputs=gr.Textbox(), title="감정 분류")
demo.launch(share = True)