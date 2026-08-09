import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import gradio as gr


classes = ['normal_RM', 'normal_TC', 'tumor_RM', 'tumor_TC']

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model = models.mobilenet_v2(pretrained=False)
model.classifier[1] = nn.Linear(model.last_channel, len(classes))

model.load_state_dict(torch.load("melhor_modelo.pth", map_location=device))
model.to(device)
model.eval()


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


def predict(image):
    image = image.convert("RGB")
    img = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        output = model(img)
        probs = torch.softmax(output, dim=1)[0]


    result = dict(zip(classes, probs.tolist()))

    normal = result['normal_RM'] + result['normal_TC']
    tumor = result['tumor_RM'] + result['tumor_TC']

    rm = result['normal_RM'] + result['tumor_RM']
    tc = result['normal_TC'] + result['tumor_TC']

    return {
        "Sem tumor": float(normal),
        "Com tumor": float(tumor),
        "Ressonância (RM)": float(rm),
        "Tomografia (TC)": float(tc)
    }


interface = gr.Interface(
    fn=predict,
    inputs=gr.Image(type="pil"),
    outputs=gr.Label(num_top_classes=4),
    title="Detector de Tumor + Tipo de Exame"
)

interface.launch()