import numpy as np
import os
from fast_sentence_transformers import FastSentenceTransformer as Tokenizer
from annoy import AnnoyIndex
import time

class Search:
    def __init__(self, binary_path: str, items: list = []):
        self.binary_path = binary_path
        self.items = items
        self.items_length = len(items)
        self.tokenizer = Tokenizer('dccuchile/bert-base-spanish-wwm-uncased', cache_folder="./cache", device="cpu", quantize=True)
        self.dimension = 768
        self.annoy = AnnoyIndex(self.dimension, 'manhattan')

        if os.path.exists(binary_path):
            self.annoy.load(binary_path)
        else:
            # # Initialize an index - the maximum number of elements should be known beforehand
            self.__insert_items(self.items)

    def __insert_items(self, items:any):
        for i, item in enumerate(items):
            self.annoy.add_item(i, self.tokenizer.encode(item, device="cpu"))
        self.annoy.build(self.items_length)
        self.annoy.save("annoy.ann")

    def search(self, query:str, top_k:int = 3):
        embedding = self.tokenizer.encode([query], device="cpu")
        nearest_neighbors = self.annoy.get_nns_by_vector(embedding[0], top_k, include_distances=True)
        
        return nearest_neighbors
    
productos= [
    {
        "name": "Pizza Hut",
        "description": "Restaurante de comida rápida que ofrece pizzas, pastas y otros platos italianos.",
        "category": "Comida",
        "ratingValue": 4.2,
        "ratingCount": 500
    },
    {
        "name": "Burger King",
        "description": "Restaurante de comida rápida que ofrece hamburguesas, papas fritas y otros platos americanos.",
        "category": "Comida",
        "ratingValue": 4.0,
        "ratingCount": 400
    },
    {
        "name": "McDonald's",
        "description": "Restaurante de comida rápida que ofrece hamburguesas, papas fritas y otros platos americanos.",
        "category": "Comida",
        "ratingValue": 4.5,
        "ratingCount": 600
    },
    {
        "name": "KFC",
        "description": "Restaurante de comida rápida que ofrece pollo frito y otros platos americanos.",
        "category": "Comida",
        "ratingValue": 4.3,
        "ratingCount": 550
    },
    {
        "name": "Subway",
        "description": "Restaurante de comida rápida que ofrece sándwiches y ensaladas personalizables.",
        "category": "Comida",
        "ratingValue": 4.1,
        "ratingCount": 450
    },
    {
        "name": "Zara",
        "description": "Tienda de ropa que ofrece ropa de moda para hombres, mujeres y niños.",
        "category": "Ropa",
        "ratingValue": 4.4,
        "ratingCount": 800
    },
    {
        "name": "H&M",
        "description": "Tienda de ropa que ofrece ropa de moda para hombres, mujeres y niños.",
        "category": "Ropa",
        "ratingValue": 4.2,
        "ratingCount": 650
    },
    {
        "name": "Nike",
        "description": "Tienda de zapatos y ropa deportiva que ofrece productos de alta calidad para atletas.",
        "category": "Zapatos",
        "ratingValue": 4.8,
        "ratingCount": 1200
    },
    {
        "name": "Adidas",
        "description": "Tienda de zapatos y ropa deportiva que ofrece productos de alta calidad para atletas.",
        "category": "Zapatos",
        "ratingValue": 4.6,
        "ratingCount": 1000
    },
    {
        "name": "Puma",
        "description": "Tienda de zapatos y ropa deportiva que ofrece productos de alta calidad para atletas.",
        "category": "Zapatos",
        "ratingValue": 4.4,
        "ratingCount": 800
    },
    {
        "name": "Domino's Pizza",
        "description": "Restaurante de comida rápida que ofrece pizzas y otros platos italianos.",
        "category": "Comida",
        "ratingValue": 4.0,
        "ratingCount": 350
    },
    {
        "name": "Bershka",
        "description": "Tienda de ropa que ofrece ropa de moda para mujeres y hombres jóvenes.",
        "category": "Ropa",
        "ratingValue": 4.1,
        "ratingCount": 400
    },
    {
        "name": "Pull & Bear",
        "description": "Tienda de ropa que ofrece ropa de moda para mujeres y hombres jóvenes.",
        "category": "Ropa",
        "ratingValue": 4.3,
        "ratingCount": 550
    },
    {
        "name": "Vans",
        "description": "Tienda de zapatos y ropa deportiva que ofrece productos de alta calidad para skaters y surfistas.",
        "category": "Zapatos",
        "ratingValue": 4.7,
        "ratingCount": 1100
    },
    {
        "name": "Converse",
        "description": "Tienda de zapatos y ropa deportiva que ofrece productos de alta calidad para atletas y amantes de la moda.",
        "category": "Zapatos",
        "ratingValue": 4.5,
        "ratingCount": 900
    }
]

# Es esencial que el texto entero del producto se guarde de una vez en BD para no calcularlo siempre
for product in productos:
  text = ""
  for name, value in product.items():
    text += f"{str(value).lower()} "
        
  product['text'] = text

# # La clase Search recibe
# 1 - La ruta relativa de donde se encuentra el binario o el nombre del archivo que desea guardar
# 2 - Los productos, un array de textos
# 3 - Los Ids de los productos en este caso cuando pertenece a la BD
nn = Search('ttt.ann', items=[producto["text"] for producto in productos]) #


# La busqueda se hace mediante tokenizar el query y dar con el ID perteneciente al objeto global
# idx = nn.search("jacket")
start_time = time.time() 
idx = nn.search("zapatos")
print("idx: ", idx)
# for id in idx:
#     print("Most similar sentence is: ", productos[id], '\n\n'+'='*100)
end_time = time.time()

elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time} seconds")