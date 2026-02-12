import torch
import torch.nn as nn
import cv2
import numpy as np
from torchvision import transforms
from torchvision.models import resnet18, ResNet18_Weights
from PIL import Image

# -------------------------------------------------
# 1. DEVICE
# -------------------------------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# -------------------------------------------------
# 2. LOAD TRAINED MODEL
# -------------------------------------------------
weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)

# change final layer (2 classes)
model.fc = nn.Linear(model.fc.in_features, 2)
# class names (same order as ImageFolder)
class_names = ["Blight", "Healthy"]

# load trained weights
model.load_state_dict(torch.load("resnet_crop.pth", map_location=device))
model = model.to(device)
model.eval()

print("Model loaded successfully ✅")

# -------------------------------------------------
# 3. IMAGE TRANSFORM
# -------------------------------------------------
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor()
])

# -------------------------------------------------
# 4. LOAD IMAGE TO EXPLAIN
# -------------------------------------------------
img_path = "dataset/test/Blight/1.JPG"   # 🔴 CHANGE IMAGE PATH
image = Image.open(img_path).convert("RGB")

input_tensor = transform(image).unsqueeze(0).to(device)

# -------------------------------------------------
# 5. GRAD-CAM HOOKS
# -------------------------------------------------
feature_maps = None
gradients = None

def forward_hook(module, input, output):
    global feature_maps
    feature_maps = output

def backward_hook(module, grad_input, grad_output):
    global gradients
    gradients = grad_output[0]

# last convolution layer in ResNet18
target_layer = model.layer4[-1].conv2
target_layer.register_forward_hook(forward_hook)
target_layer.register_backward_hook(backward_hook)

# -------------------------------------------------
# 6. FORWARD & BACKWARD PASS
# -------------------------------------------------
output = model(input_tensor)
predicted_class = output.argmax(dim=1).item()

print("Predicted class:", class_names[predicted_class])


model.zero_grad()
output[0, predicted_class].backward()

# -------------------------------------------------
# 7. COMPUTE GRAD-CAM HEATMAP
# -------------------------------------------------
# gradients & feature maps shape: [1, C, H, W]
weights = gradients.mean(dim=(1, 2))  # importance of each feature map

cam = torch.zeros(feature_maps.shape[2:], device=device)

for i, w in enumerate(weights):
    cam += w * feature_maps[0, i, :, :]

cam = torch.relu(cam)
cam -= cam.min()
cam /= cam.max()

cam = cam.cpu().detach().numpy()

# -------------------------------------------------
# 8. OVERLAY HEATMAP ON IMAGE
# -------------------------------------------------
# Resize heatmap
cam = cv2.resize(cam, (224, 224))
heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)

original_img = np.array(image.resize((224, 224)))
overlay = cv2.addWeighted(original_img, 0.6, heatmap, 0.4, 0)

# ---------------- SAVE IMAGE ----------------
output_path = "gradcam_result.jpg"
cv2.imwrite(output_path, overlay)

print("Grad-CAM image saved as gradcam_result.jpg ✅")

# ---------------- SHOW IMAGE ----------------
cv2.imshow("Grad-CAM Result", overlay)
cv2.waitKey(0)
cv2.destroyAllWindows()

