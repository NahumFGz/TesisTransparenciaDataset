import copy
import os
from collections import defaultdict

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, models, transforms
from tqdm import tqdm


# ---------------------------------------------------------
# 1) Función para usar solo un porcentaje subset_ratio de cada clase
#    Mantiene balance de clases.
# ---------------------------------------------------------
def subset_classwise(dataset, subset_ratio=1.0):
    """
    Retorna un Subset de 'dataset' con la proporción 'subset_ratio' de
    cada clase de manera uniforme.

    Ej: subset_ratio=0.1 -> 10% de cada clase.
    """
    if subset_ratio >= 1.0:
        return dataset  # Usa 100% de los datos

    # Agrupamos índices de dataset.samples por clase
    class_indices = defaultdict(list)
    for idx, (_, class_idx) in enumerate(dataset.samples):
        class_indices[class_idx].append(idx)

    # Construimos una lista de los índices que vamos a quedarnos
    selected_indices = []
    for class_idx, idxs in class_indices.items():
        keep_count = int(len(idxs) * subset_ratio)
        selected_indices.extend(idxs[:keep_count])  # tomamos subset

    return Subset(dataset, selected_indices)


# ---------------------------------------------------------
# 2) Obtener dimensiones de entrada según el modelo
# ---------------------------------------------------------
def get_input_size(model_name: str):
    model_name = model_name.lower()
    if model_name in ["resnet50", "vgg16", "densenet121", "mobilenet_v2", "efficientnet_b0", "googlenet"]:
        return (224, 224)
    elif model_name == "inception_v3":
        return (299, 299)
    else:
        return (224, 224)  # default


# ---------------------------------------------------------
# 3) Obtener el "cuerpo" (feature extractor) sin la FC original
#    y el número de canales (out_features) que salen de ese cuerpo.
# ---------------------------------------------------------
def get_base_model(model_name: str):
    model_name = model_name.lower()

    if model_name == "resnet50":
        base = models.resnet50(weights=models.ResNet50_Weights.DEFAULT)
        # Guardamos el número de features que salen de layer final
        out_features = base.fc.in_features
        # Quitamos la capa fc, para luego reemplazarla
        base.fc = nn.Identity()  # No hace nada, deja pasar el tensor
        return base, out_features

    elif model_name == "mobilenet_v2":
        base = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # out_features de la parte conv + pool final
        out_features = base.classifier[1].in_features
        # Quitamos la parte classifier final, para luego añadir nuestra top
        base.classifier = nn.Identity()
        return base, out_features

    else:
        raise NotImplementedError(f"No implementado para '{model_name}' aún.")


# ---------------------------------------------------------
# 4) TopModelBlock: análogo a lo que hacías con Keras (top_model=1/2)
# ---------------------------------------------------------
class TopModelBlock(nn.Module):
    def __init__(self, top_model: int, in_features: int, num_classes: int):
        super().__init__()
        self.top_model = top_model

        if top_model == 1:
            # "GlobalAveragePooling2D" + 1 Dense softmax
            # En PyTorch: usaremos AdaptiveAvgPool2d para emular GlobalAvgPool2d
            #  -> flatten
            #  -> Dense(num_classes)
            self.gap = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(in_features, num_classes)

        elif top_model == 2:
            # Flatten -> Dense(256, relu) -> Dropout(0.4)
            #         -> Dense(128, relu) -> Dropout(0.4)
            #         -> Dense(num_classes, softmax)
            self.flatten = nn.Flatten()  # aplanar
            self.dense1 = nn.Linear(in_features, 256)
            self.dropout1 = nn.Dropout(0.4)
            self.dense2 = nn.Linear(256, 128)
            self.dropout2 = nn.Dropout(0.4)
            self.fc = nn.Linear(128, num_classes)
        else:
            raise ValueError("top_model debe ser 1 o 2")

    def forward(self, x):
        if self.top_model == 1:
            # GAP
            x = self.gap(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
        elif self.top_model == 2:
            x = self.flatten(x)
            x = nn.functional.relu(self.dense1(x))
            x = self.dropout1(x)
            x = nn.functional.relu(self.dense2(x))
            x = self.dropout2(x)
            x = self.fc(x)
        return x


# ---------------------------------------------------------
# 5) CustomNet que une el base_model (feature extractor) + top_model
# ---------------------------------------------------------
class CustomNet(nn.Module):
    def __init__(self, base_model: nn.Module, top_block: nn.Module):
        super().__init__()
        self.base_model = base_model
        self.top_block = top_block

    def forward(self, x):
        # 1) Extraer features del modelo base
        x = self.base_model(x)
        # 2) Pasar por la cabeza
        x = self.top_block(x)
        return x


# ---------------------------------------------------------
# 6) Congelar capas hasta 'block_layer_number'
#    Cada arquitectura se maneja diferente; aquí
#    tenemos un ejemplo para ResNet y MobileNetV2
# ---------------------------------------------------------
def freeze_until_layer(model: nn.Module, model_name: str, block_layer_number: int):
    """
    Congela (requires_grad=False) hasta cierta capa/bloque.
    - En ResNet50: children -> [conv1, bn1, relu, maxpool, layer1, layer2, ...]
    - En MobileNetV2: model.features es un contenedor con ~19 bloques
    Ajustar block_layer_number según tu necesidad.
    """
    name = model_name.lower()

    if "resnet" in name:
        children = list(model.children())  # conv1, bn1, relu, maxpool, layer1...
        for idx, child in enumerate(children):
            if idx < block_layer_number:
                for param in child.parameters():
                    param.requires_grad = False
            else:
                for param in child.parameters():
                    param.requires_grad = True

    elif "mobilenet_v2" in name:
        # MobileNetV2 -> .features es un Sequential con sub-bloques
        if hasattr(model, "features"):
            layers = list(model.features.children())
            for i, layer in enumerate(layers):
                if i < block_layer_number:
                    for param in layer.parameters():
                        param.requires_grad = False
                else:
                    for param in layer.parameters():
                        param.requires_grad = True
        # la parte classifier (si existiera) la podemos dejar entrenable
        # o congelarla si deseamos, pero usualmente la dejamos entrenable.

    else:
        print(f"freeze_until_layer() no implementado para {model_name}.")


# ---------------------------------------------------------
# 7) Cargar DataLoaders
# ---------------------------------------------------------
def get_dataloaders(
    data_dir: str, model_name: str, batch_size: int, split_ratio: float = 0.8, subset_ratio: float = 1.0
):
    input_size = get_input_size(model_name)

    transform = transforms.Compose(
        [
            transforms.Resize(input_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    # Cargar dataset completo
    dataset = datasets.ImageFolder(root=data_dir, transform=transform)

    # Filtrar por subset_ratio si es < 1.0
    dataset = subset_classwise(dataset, subset_ratio=subset_ratio)

    # Split train/val
    train_size = int(len(dataset) * split_ratio)
    val_size = len(dataset) - train_size
    train_set, val_set = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)

    num_classes = len(dataset.dataset.classes) if isinstance(dataset, Subset) else len(dataset.classes)

    return train_loader, val_loader, num_classes


# ---------------------------------------------------------
# 8) Función principal de entrenamiento
# ---------------------------------------------------------
def train_model(
    model_name: str,
    data_dir: str,
    top_model: int = 1,
    block_layer_number: int = 0,
    batch_size: int = 32,
    num_epochs: int = 10,
    lr: float = 1e-4,
    split_ratio: float = 0.8,
    subset_ratio: float = 1.0,
    device: torch.device = None,
):
    """
    - model_name: ResNet50, MobileNet_V2, etc.
    - top_model: 1 (GAP->Dense) o 2 (Flatten->Dense(256)->Dropout...).
    - block_layer_number: cuántos bloques congelar (depende de la arquitectura).
    - subset_ratio: porcentaje de datos (por clase) para entrenar (ej: 0.1 -> 10%).
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Usando dispositivo: {device}")

    # 1) Dataloaders
    train_loader, val_loader, num_classes = get_dataloaders(
        data_dir=data_dir,
        model_name=model_name,
        batch_size=batch_size,
        split_ratio=split_ratio,
        subset_ratio=subset_ratio,
    )

    # 2) Modelo base y top
    base_model, out_features = get_base_model(model_name)
    top_block = TopModelBlock(top_model=top_model, in_features=out_features, num_classes=num_classes)

    # 3) Combine en un solo modelo
    model = CustomNet(base_model, top_block).to(device)

    # 4) Congelar capas
    if block_layer_number > 0:
        freeze_until_layer(base_model, model_name, block_layer_number)

    # 5) Definir pérdida y optimizador
    #    (filtramos solo parámetros con requires_grad=True)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    # 6) Entrenamiento
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f"\n📦 Epoch {epoch+1}/{num_epochs}")
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
                dataloader = train_loader
            else:
                model.eval()
                dataloader = val_loader

            running_loss = 0.0
            running_corrects = 0

            loop = tqdm(dataloader, desc=f"{phase.capitalize()} Phase")
            for inputs, labels in loop:
                inputs, labels = inputs.to(device), labels.to(device)

                with torch.set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    if phase == "train":
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloader.dataset)
            epoch_acc = running_corrects.double() / len(dataloader.dataset)

            print(f"{phase} Loss: {epoch_loss:.4f} | Acc: {epoch_acc:.4f}")

            # Guardar mejor modelo
            if phase == "val" and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

    print(f"\n✅ Mejor accuracy en validación: {best_acc:.4f}")

    # 7) Cargar mejores pesos y guardar
    model.load_state_dict(best_model_wts)
    save_name = f"{model_name}_top{top_model}_best.pth"
    torch.save(model.state_dict(), save_name)
    print(f"Modelo guardado como: {save_name}")

    return model


# ---------------------------------------------------------
# 9) Ejemplo de uso
# ---------------------------------------------------------
if __name__ == "__main__":
    MODEL_NAME = "resnet50"  # "resnet50" o "mobilenet_v2", etc.
    DATA_DIR = "/path/to/dataset"  # Ajusta tu ruta
    TOP_MODEL = 2  # 1 o 2 (ver la clase TopModelBlock)
    BLOCK_LAYER_NUMBER = 4  # Congelar las primeras 4 "child layers" (depende de la arquitectura)
    BATCH_SIZE = 32
    NUM_EPOCHS = 5
    LR = 1e-4
    SPLIT_RATIO = 0.8  # 80% train, 20% val
    SUBSET_RATIO = 0.5  # Usar solo 50% de datos en cada clase

    trained_model = train_model(
        model_name=MODEL_NAME,
        data_dir=DATA_DIR,
        top_model=TOP_MODEL,
        block_layer_number=BLOCK_LAYER_NUMBER,
        batch_size=BATCH_SIZE,
        num_epochs=NUM_EPOCHS,
        lr=LR,
        split_ratio=SPLIT_RATIO,
        subset_ratio=SUBSET_RATIO,
    )
