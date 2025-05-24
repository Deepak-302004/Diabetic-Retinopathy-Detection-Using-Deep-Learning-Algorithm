from flask import Flask, render_template, request, redirect, url_for, session, flash
from PIL import Image
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
from werkzeug.utils import secure_filename
import os
import numpy as np
import cv2
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import A4
from datetime import datetime
from reportlab.lib.pagesizes import landscape, A4
from reportlab.platypus import Table, TableStyle
from reportlab.lib import colors

app = Flask(__name__)
app.secret_key = 'your_secret_key'
app.config['SESSION_TYPE'] = 'filesystem'

# Simulated login user
users = {'admin': 'password'}

# Set device
device = 'cuda' if torch.cuda.is_available() else 'cpu'
class_names = ['Mild', 'Moderate', 'No_DR', 'Proliferate_DR', 'Severe']

# Custom Model
class Block(nn.Module):
    expansion = 2
    def __init__(self, in_planes, cardinality=32, bottleneck_width=4, stride=1):
        super(Block, self).__init__()
        group_width = cardinality * bottleneck_width
        self.conv1 = nn.Conv2d(in_planes, group_width, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(group_width)
        self.conv2 = nn.Conv2d(group_width, group_width, kernel_size=3, stride=stride, padding=1, groups=cardinality, bias=False)
        self.bn2 = nn.BatchNorm2d(group_width)
        self.conv3 = nn.Conv2d(group_width, self.expansion*group_width, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion*group_width)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*group_width:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*group_width, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion*group_width)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class EfficientnetB0(nn.Module):
    def __init__(self, num_blocks, cardinality, bottleneck_width, num_classes=5):
        super(EfficientnetB0, self).__init__()
        self.cardinality = cardinality
        self.bottleneck_width = bottleneck_width
        self.in_planes = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(num_blocks[0], 1)
        self.layer2 = self._make_layer(num_blocks[1], 2)
        self.layer3 = self._make_layer(num_blocks[2], 2)
        self.linear = nn.Linear(cardinality * bottleneck_width * 8, num_classes)

    def _make_layer(self, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(Block(self.in_planes, self.cardinality, self.bottleneck_width, stride))
            self.in_planes = Block.expansion * self.cardinality * self.bottleneck_width
        self.bottleneck_width *= 2
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        self.feature_maps = out  # store for Grad-CAM
        out = F.avg_pool2d(out, 6)
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        return out

# Load model
model = EfficientnetB0(num_blocks=[3, 3, 3], cardinality=32, bottleneck_width=4)
model.load_state_dict(torch.load('ckpts1.pth', map_location=device)['net'])
model.to(device)
model.eval()

# Transforms
transform = transforms.Compose([
    transforms.Resize(24),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

# Folders
UPLOAD_FOLDER = os.path.join('static', 'uploads')
GRADCAM_FOLDER = os.path.join('static', 'gradcam')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(GRADCAM_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def generate_gradcam_plus_plus(input_tensor, output, pred_class, model, image_path):
    model.zero_grad()
    one_hot_output = torch.zeros_like(output)
    one_hot_output[0][pred_class] = 1
    output.backward(gradient=one_hot_output)

    gradients = model.feature_maps.grad
    activations = model.feature_maps

    with torch.no_grad():
        grads = gradients[0]
        acts = activations[0]

        alpha_num = grads.pow(2)
        alpha_denom = grads.pow(2).mul(2) + acts.mul(grads.pow(3)).sum(dim=(1, 2), keepdim=True)
        alpha_denom = torch.where(alpha_denom != 0.0, alpha_denom, torch.ones_like(alpha_denom))
        alphas = alpha_num / alpha_denom
        weights = (alphas * F.relu(grads)).sum(dim=(1, 2))

        heatmap = torch.sum(weights[:, None, None] * acts, dim=0)
        heatmap = F.relu(heatmap)
        heatmap -= heatmap.min()
        heatmap /= heatmap.max()

        heatmap = heatmap.cpu().numpy()

    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    heatmap = cv2.resize(heatmap, (img.shape[1], img.shape[0]))
    heatmap_color = cv2.applyColorMap(np.uint8(255 * heatmap), cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(img, 0.6, heatmap_color, 0.4, 0)

    gradcam_path = os.path.join(GRADCAM_FOLDER, os.path.basename(image_path))
    cv2.imwrite(gradcam_path, cv2.cvtColor(superimposed_img, cv2.COLOR_RGB2BGR))
    return gradcam_path

def generate_pdf_report(input_image_path, gradcam_image_path, prediction, confidence, class_probs, output_path):
    c = canvas.Canvas(output_path, pagesize=landscape(A4))
    width, height = landscape(A4)

    # 🏥 Branding Header
    c.setFont("Helvetica-Bold", 20)
    c.drawString(50, height - 50, "■ AI Diagnostics System")

    # 🧾 Report Title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 90, "Diabetic Retinopathy Prediction Report")

    # ✨ Details
    c.setFont("Helvetica", 12)
    c.drawString(50, height - 130, f"Prediction: {prediction}")
    c.drawString(50, height - 150, f"Confidence: {confidence}%")
    c.drawString(50, height - 170, f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # 📊 Class Probabilities Table
    c.setFont("Helvetica-Bold", 12)
    c.drawString(50, height - 210, "Class Probabilities (%):")

    table_data = [["Class", "Probability (%)"]] + [[cls, f"{prob}"] for cls, prob in class_probs]
    table = Table(table_data, colWidths=[200, 150])
    table.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), colors.lightblue),
        ("TEXTCOLOR", (0, 0), (-1, 0), colors.black),
        ("ALIGN", (0, 0), (-1, -1), "CENTER"),
        ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
        ("BOTTOMPADDING", (0, 0), (-1, 0), 12),
        ("BACKGROUND", (0, 1), (-1, -1), colors.whitesmoke),
        ("GRID", (0, 0), (-1, -1), 1, colors.black),
    ]))
    
    # Wrap and draw the table
    table_width, table_height = table.wrapOn(c, width, height)
    table_x = 50
    table_y = height - 250 - table_height
    table.drawOn(c, table_x, table_y)

    # 📸 Draw Input & Grad-CAM Images just below the table
    image_y = table_y - 180  # Adjust vertical spacing from table to images

    c.setFont("Helvetica-Bold", 12)
    c.drawString(200, image_y + 160, "Input Image:")
    c.drawImage(input_image_path, 100, image_y, width=250, height=150, preserveAspectRatio=True)

    c.drawString(530, image_y + 160, "Grad-CAM Output:")
    c.drawImage(gradcam_image_path, 450, image_y, width=250, height=150, preserveAspectRatio=True)

    c.save()

# Routes
@app.route('/')
def index():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        uname = request.form['username']
        pwd = request.form['password']
        if uname in users and users[uname] == pwd:
            session['username'] = uname
            return redirect(url_for('index'))
        else:
            flash("Invalid Credentials")
    return render_template('login.html')

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('login'))

@app.route('/predict', methods=['POST'])
def predict():
    if 'username' not in session:
        return redirect(url_for('login'))

    if 'file' not in request.files or request.files['file'].filename == '':
        flash("No file uploaded")
        return redirect(request.url)

    file = request.files['file']
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    # Load image
    image = Image.open(filepath).convert('RGB')
    input_tensor = transform(image).unsqueeze(0).to(device).requires_grad_()

    model.zero_grad()
    outputs = model(input_tensor)
    _, pred_class = outputs.max(1)
    prediction = class_names[pred_class.item()]

    # Calculate confidence score
    probabilities = F.softmax(outputs, dim=1)[0]  # Get the first sample's probs
    confidence_score = torch.max(probabilities).item() * 100
    confidence_score = round(confidence_score, 2)

    # Zip class names and probs into a list of tuples
    class_probabilities = [(class_names[i], round(prob.item() * 100, 2)) for i, prob in enumerate(probabilities)]


    # Grad-CAM++
    model.feature_maps.retain_grad()
    gradcam_path = generate_gradcam_plus_plus(input_tensor, outputs, pred_class.item(), model, filepath)

    gradcam_url = url_for('static', filename='gradcam/' + filename)
    input_img_url = url_for('static', filename='uploads/' + filename)
    
    # Safe filename without extension for use
    clean_filename = os.path.splitext(filename)[0]
    pdf_filename = f"report_{clean_filename}.pdf"
    pdf_output_path = os.path.join('static', 'reports', pdf_filename)

    # Ensure directory exists
    os.makedirs(os.path.dirname(pdf_output_path), exist_ok=True)

    # Generate the PDF report
    generate_pdf_report(filepath, gradcam_path, prediction, confidence_score, class_probabilities, pdf_output_path)

    # Final link for download
    pdf_url = url_for('static', filename=f'reports/{pdf_filename}')

    
    return render_template('result.html',
    prediction=prediction,
    confidence=confidence_score,
    gradcam_image=gradcam_url,
    input_image=input_img_url,
    class_probs=class_probabilities,
    pdf_filename=pdf_url  # pass full URL
)


@app.route('/about')
def about():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('about.html')

@app.route('/contact')
def contact():
    if 'username' not in session:
        return redirect(url_for('login'))
    return render_template('contact.html')

if __name__ == '__main__':
    app.run(debug=True)