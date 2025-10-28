import os, glob
DIR = "./augmented"
imgs = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(DIR,"*.jpg"))+glob.glob(os.path.join(DIR,"*.png"))}
xmls = {os.path.splitext(os.path.basename(p))[0] for p in glob.glob(os.path.join(DIR,"*.xml"))}

solo_imgs = sorted(imgs - xmls)
solo_xmls = sorted(xmls - imgs)

print(f"Total archivos: {len(imgs)*2} (si todo pareado)")
print(f"Imágenes: {len(imgs)} | XML: {len(xmls)}")
print(f"Sin pareja (solo imagen): {len(solo_imgs)}")
print(f"Sin pareja (solo xml): {len(solo_xmls)}")
if solo_imgs[:10]: print("Ejemplos solo imagen:", solo_imgs[:10])
if solo_xmls[:10]: print("Ejemplos solo xml:", solo_xmls[:10])
