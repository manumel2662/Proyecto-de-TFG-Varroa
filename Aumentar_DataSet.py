import os, glob, random, shutil
import xml.etree.ElementTree as ET
from xml.dom import minidom
import cv2
import numpy as np
from tqdm import tqdm
import albumentations as A

# === CONFIG ===
INPUT_DIR = "./DatasetOriginal"     # carpeta con JPG/PNG + XML
OUTPUT_DIR = "./augmented"          # salida sin tocar el original
AUGS_PER_IMAGE = 3                  # x3: triplicas dataset (ajústalo a 2 o 3)
SEED = 42
random.seed(SEED); np.random.seed(SEED)

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ---- PIPELINE DE AUGMENTACIÓN (ajustada) ----
transform = A.Compose([
    A.OneOf([
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.2),
        A.RandomRotate90(p=0.3)
    ], p=0.7),

    A.ShiftScaleRotate(shift_limit=0.02, scale_limit=0.05, rotate_limit=8,
                       border_mode=cv2.BORDER_REFLECT_101, p=0.7),

    A.OneOf([
        A.RandomBrightnessContrast(brightness_limit=0.15, contrast_limit=0.15, p=0.7),
        A.CLAHE(clip_limit=2.0, tile_grid_size=(8,8), p=0.3)
    ], p=0.8),

    A.OneOf([
        A.MotionBlur(p=0.2),
        A.MedianBlur(blur_limit=3, p=0.2),
        A.GaussianBlur(blur_limit=3, p=0.2),
        # Rango normalizado para evitar el error de validación:
        A.GaussNoise(var_limit=(0.001, 0.01), mean=0, per_channel=True, p=0.2),
    ], p=0.4),
], bbox_params=A.BboxParams(format='pascal_voc', label_fields=['labels'],
                            min_area=4, min_visibility=0.2))

# ---- LECTURA DE VOC CON "CLAMP" EN CAJAS ----
def read_voc(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    W = int(root.find('size/width').text)
    H = int(root.find('size/height').text)

    def clamp(v, lo, hi):
        return max(lo, min(hi, v))

    boxes, labels = [], []
    for obj in root.findall('object'):
        name = obj.find('name').text.strip().lower()
        if name.startswith("varr"):
            name = "varroa"

        bb = obj.find('bndbox')
        xmin = int(float(bb.find('xmin').text))
        ymin = int(float(bb.find('ymin').text))
        xmax = int(float(bb.find('xmax').text))
        ymax = int(float(bb.find('ymax').text))

        # Limitar a 0..W-1 / 0..H-1 y corregir invertidas
        xmin = clamp(xmin, 0, W-1); xmax = clamp(xmax, 0, W-1)
        ymin = clamp(ymin, 0, H-1); ymax = clamp(ymax, 0, H-1)
        if xmax < xmin: xmin, xmax = xmax, xmin
        if ymax < ymin: ymin, ymax = ymax, ymin

        if (xmax - xmin) >= 2 and (ymax - ymin) >= 2:
            boxes.append([xmin, ymin, xmax, ymax])
            labels.append(name)

    return tree, (W, H), boxes, labels

# ---- ESCRITURA VOC ----
def write_voc(template_tree, img_name, out_xml_path, bboxes, labels, width, height):
    root = template_tree.getroot()
    root.find('filename').text = img_name
    path_el = root.find('path')
    if path_el is not None:
        path_el.text = img_name

    size = root.find('size')
    size.find('width').text = str(width)
    size.find('height').text = str(height)
    size.find('depth').text = "3"

    # borrar objetos antiguos
    for obj in root.findall('object'):
        root.remove(obj)

    # añadir nuevos
    for (xmin, ymin, xmax, ymax), label in zip(bboxes, labels):
        obj = ET.SubElement(root, 'object')
        ET.SubElement(obj, 'name').text = label
        ET.SubElement(obj, 'pose').text = 'Unspecified'
        ET.SubElement(obj, 'truncated').text = '0'
        ET.SubElement(obj, 'difficult').text = '0'
        ET.SubElement(obj, 'occluded').text = '0'
        bnd = ET.SubElement(obj, 'bndbox')
        ET.SubElement(bnd, 'xmin').text = str(int(max(0, xmin)))
        ET.SubElement(bnd, 'ymin').text = str(int(max(0, ymin)))
        ET.SubElement(bnd, 'xmax').text = str(int(min(width-1, xmax)))
        ET.SubElement(bnd, 'ymax').text = str(int(min(height-1, ymax)))

    xml_str = ET.tostring(root, encoding='utf-8')
    with open(out_xml_path, 'w', encoding='utf-8') as f:
        f.write(minidom.parseString(xml_str).toprettyxml(indent="\t"))

def safe_name(base, idx):
    stem, ext = os.path.splitext(base)
    return f"{stem}_aug{idx}{ext}"

# ---- PROCESO ----
images = sorted(glob.glob(os.path.join(INPUT_DIR, "*.jpg")) +
                glob.glob(os.path.join(INPUT_DIR, "*.png")))
assert images, f"No se encontraron imágenes en {INPUT_DIR}"

for img_path in tqdm(images, desc="Augmentando"):
    xml_path = os.path.splitext(img_path)[0] + ".xml"
    if not os.path.exists(xml_path):
        shutil.copy2(img_path, os.path.join(OUTPUT_DIR, os.path.basename(img_path)))
        continue

    tree, (W, H), boxes, labels = read_voc(xml_path)
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Copia ORIGINAL a OUTPUT (comodidad)
    base = os.path.basename(img_path)
    out_img0 = os.path.join(OUTPUT_DIR, base)
    cv2.imwrite(out_img0, img)
    write_voc(tree, os.path.basename(out_img0),
              os.path.join(OUTPUT_DIR, os.path.splitext(base)[0] + ".xml"),
              boxes, labels, W, H)

    # Augmentaciones
    for i in range(1, AUGS_PER_IMAGE + 1):
        tried = 0
    while tried < 6:
        tried += 1
        t = transform(image=img, bboxes=boxes, labels=labels)

        tb = []
        for (xmin, ymin, xmax, ymax) in t['bboxes']:
            # Limitar las coordenadas al tamaño de la imagen
            xmin = max(0, min(W-1, int(xmin)))
            xmax = max(0, min(W-1, int(xmax)))
            ymin = max(0, min(H-1, int(ymin)))
            ymax = max(0, min(H-1, int(ymax)))

            # 🔹 Filtro 1: descartar si toca el borde
            if xmin <= 1 or ymin <= 1 or xmax >= W-2 or ymax >= H-2:
                continue  # evita cajas pegadas o recortadas en los bordes

            # 🔹 Filtro 2: descartar cajas demasiado pequeñas
            if (xmax - xmin) < 4 or (ymax - ymin) < 4:
                continue

            tb.append([xmin, ymin, xmax, ymax])

        tl = t['labels']

        # Si después de filtrar quedan cajas válidas
        if tb:
            out_img_name = safe_name(base, i)
            out_img_path = os.path.join(OUTPUT_DIR, out_img_name)
            cv2.imwrite(out_img_path, t['image'])

            out_xml_path = os.path.join(OUTPUT_DIR, os.path.splitext(out_img_name)[0] + ".xml")
            write_voc(tree, out_img_name, out_xml_path, tb, tl, W, H)
            break  # esta augmentación salió bien, pasar a la siguiente

