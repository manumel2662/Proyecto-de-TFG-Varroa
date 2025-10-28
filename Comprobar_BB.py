import os, glob, random, xml.etree.ElementTree as ET
import cv2

IN_DIR = "./augmented"         # carpeta con JPG/PNG + XML tras la augmentación
OUT_DIR = "./debug_bb"         # salidas con bboxes dibujados
N_SAMPLES = 40                 # cuántas muestras quieres revisar
os.makedirs(OUT_DIR, exist_ok=True)

def read_boxes(xml_path):
    root = ET.parse(xml_path).getroot()
    W = int(root.find('size/width').text); H = int(root.find('size/height').text)
    boxes = []
    for obj in root.findall('object'):
        bb = obj.find('bndbox')
        xmin = int(float(bb.find('xmin').text))
        ymin = int(float(bb.find('ymin').text))
        xmax = int(float(bb.find('xmax').text))
        ymax = int(float(bb.find('ymax').text))
        # clamp por si acaso
        xmin = max(0, min(W-1, xmin)); xmax = max(0, min(W-1, xmax))
        ymin = max(0, min(H-1, ymin)); ymax = max(0, min(H-1, ymax))
        if xmax > xmin and ymax > ymin:
            boxes.append((xmin, ymin, xmax, ymax))
    return boxes

imgs = sorted(glob.glob(os.path.join(IN_DIR, "*.jpg")) + glob.glob(os.path.join(IN_DIR, "*.png")))
random.seed(42)
samples = random.sample(imgs, min(N_SAMPLES, len(imgs)))

for ip in samples:
    xp = os.path.splitext(ip)[0] + ".xml"
    if not os.path.exists(xp): continue
    im = cv2.imread(ip)
    if im is None: continue
    for (xmin,ymin,xmax,ymax) in read_boxes(xp):
        cv2.rectangle(im, (xmin,ymin), (xmax,ymax), (0,255,0), 2)  # dibuja caja
    out = os.path.join(OUT_DIR, os.path.basename(ip))
    cv2.imwrite(out, im)

print(f"Listo. Revisa {OUT_DIR} con {len(samples)} imágenes de muestra.")
