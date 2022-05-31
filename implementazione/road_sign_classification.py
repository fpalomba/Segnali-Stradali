import bs4
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image, ImageTk
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

tf.random.set_seed(100)

path = "..\\resources\\annotations"
content = []
speedcounter = 0

for filename in os.listdir(path):

    if not filename.endswith('.xml'): continue
    finalpath = os.path.join(path, filename)

    infile = open(finalpath, "r")

    contents = infile.read()
    # Usiamo la libreria BeautifulSoup per estrarre i dati dal file xml(altezza,larghezza e profondità dell'immagine)
    soup = bs4.BeautifulSoup(contents, 'xml')
    class_name = soup.find_all("name")
    name = soup.find_all('filename')
    width = soup.find_all("width")
    height = soup.find_all("height")
    depth = soup.find_all("depth")

    ls = []
    for x in range(0, len(name)):
        for i in name:
            name = name[x].get_text()
            path_name = "images/" + name

        class_name = class_name[x].get_text()
        if class_name == 'speedlimit':
            if speedcounter < 75:
                # Counter per le immagini della classe 'limiti di velocità'
                height = int(height[x].get_text())
                depth = int(depth[x].get_text())
                width = int(width[x].get_text())
                f_name = filename
                ls.extend([f_name, path_name, width, height, depth, class_name])
                speedcounter = speedcounter + 1
                content.append(ls)
        else:
            # Seleziona le altre classi di immagini(segnali di stop,attraversamenti pedonali,semafori)
            height = int(height[x].get_text())
            depth = int(depth[x].get_text())
            width = int(width[x].get_text())
            f_name = filename
            ls.extend([f_name, path_name, width, height, depth, class_name])
            content.append(ls)

new_cols = ["f_name", "path_name", "width", "height", "depth", "class_name"]
data = pd.DataFrame(data=content, columns=new_cols)
data.class_name = data.class_name.map({'trafficlight': 1, 'speedlimit': 2, 'crosswalk': 3, 'stop': 4})
print(data.shape)