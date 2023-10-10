from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer
import pandas as pd
import lancedb
from lancedb.embeddings import with_embeddings
import time

class LanceDB:
    def __init__(self, table_name: str, data: list[any] = None, create_table:bool=False):
        self.model_name = 'dccuchile/bert-base-spanish-wwm-uncased' # modelo para usar en el tokenizado
        self.db = lancedb.connect("./.lancedb") # conneccion a la db local
        self.table_name = table_name # nombre de la tabla
        self.table_opened = None # validacion de la tabla activa
        self.data = data # data para ingestar o inferir

        if create_table: # util para crear una tabla nueva
            if data is None: raise ValueError("Create table needs data items list")
            model = SentenceTransformer(self.model_name, cache_folder="./cache", device="cpu", quantize=True)
            payload = with_embeddings(self.embed_func, self.data)
            self.table_opened = db.create_table(self.table_name, payload, mode="overwrite")
            self.table_opened.create_fts_index("text")
        else:
            try:
                self.table_opened = self.db.open_table(self.table_name) # en caso ya exista la tabla
            except:
                raise ValueError("Table '%s' not found" % self.table_name)
        
    def __embed_func(self, data):
        '''
            @params
                data: list of objects wich contains metadata about the item
            @returns: the embeddings needs to be load data into lanceDb
        '''
        data = pd.DataFrame(data)
        return [model.encode(item, device="cpu") for item in data]

    def search(self, query: str, to_vector:bool=False, limit: int = 5, select: list[str] = ["text"]):
        '''
            @params
                query: str
                to_vector: bool by default False
                limit: int by default 5
                select: list[str] by default ["text]
            @returns: a DataFrame containing the results or empty DataFrame
        '''
        if self.table_opened == None: raise ValueError("Table '%s' not found" % self.table_name)
        if to_vector:
            model = SentenceTransformer(self.model_name, cache_folder="./cache", device="cpu", quantize=True)
            query = model.encode([query], device="cpu")[0]
        return self.table_opened.search(query).limit(limit).select(select).to_df()



# products= [
#     {
#         "name": "Pizza Hut",
#         "description": "Restaurante de comida rápida que ofrece pizzas, pastas y otros platos italianos.",
#         "category": "Comida",
#         "ratingValue": 4.2,
#         "ratingCount": 500
#     },
#     {
#         "name": "Burger King",
#         "description": "Restaurante de comida rápida que ofrece hamburguesas, papas fritas y otros platos americanos.",
#         "category": "Comida",
#         "ratingValue": 4.0,
#         "ratingCount": 400
#     },
#     {
#         "name": "McDonald's",
#         "description": "Restaurante de comida rápida que ofrece hamburguesas, papas fritas y otros platos americanos.",
#         "category": "Comida",
#         "ratingValue": 4.5,
#         "ratingCount": 600
#     },
#     {
#         "name": "KFC",
#         "description": "Restaurante de comida rápida que ofrece pollo frito y otros platos americanos.",
#         "category": "Comida",
#         "ratingValue": 4.3,
#         "ratingCount": 550
#     },
#     {
#         "name": "Subway",
#         "description": "Restaurante de comida rápida que ofrece sándwiches y ensaladas personalizables.",
#         "category": "Comida",
#         "ratingValue": 4.1,
#         "ratingCount": 450
#     },
#     {
#         "name": "Zara",
#         "description": "Tienda de ropa que ofrece ropa de moda para hombres, mujeres y niños.",
#         "category": "Ropa",
#         "ratingValue": 4.4,
#         "ratingCount": 800
#     },
#     {
#         "name": "H&M",
#         "description": "Tienda de ropa que ofrece ropa de moda para hombres, mujeres y niños.",
#         "category": "Ropa",
#         "ratingValue": 4.2,
#         "ratingCount": 650
#     },
#     {
#         "name": "Nike",
#         "description": "Tienda de zapatos y ropa deportiva que ofrece productos de alta calidad para atletas.",
#         "category": "Zapatos",
#         "ratingValue": 4.8,
#         "ratingCount": 1200
#     },
#     {
#         "name": "Adidas",
#         "description": "Tienda de zapatos y ropa deportiva que ofrece productos de alta calidad para atletas.",
#         "category": "Zapatos",
#         "ratingValue": 4.6,
#         "ratingCount": 1000
#     },
#     {
#         "name": "Puma",
#         "description": "Tienda de zapatos y ropa deportiva que ofrece productos de alta calidad para atletas.",
#         "category": "Zapatos",
#         "ratingValue": 4.4,
#         "ratingCount": 800
#     },
#     {
#         "name": "Domino's Pizza",
#         "description": "Restaurante de comida rápida que ofrece pizzas y otros platos italianos.",
#         "category": "Comida",
#         "ratingValue": 4.0,
#         "ratingCount": 350
#     },
#     {
#         "name": "Bershka",
#         "description": "Tienda de ropa que ofrece ropa de moda para mujeres y hombres jóvenes.",
#         "category": "Ropa",
#         "ratingValue": 4.1,
#         "ratingCount": 400
#     },
#     {
#         "name": "Pull & Bear",
#         "description": "Tienda de ropa que ofrece ropa de moda para mujeres y hombres jóvenes.",
#         "category": "Ropa",
#         "ratingValue": 4.3,
#         "ratingCount": 550
#     },
#     {
#         "name": "Vans",
#         "description": "Tienda de zapatos y ropa deportiva que ofrece productos de alta calidad para skaters y surfistas.",
#         "category": "Zapatos",
#         "ratingValue": 4.7,
#         "ratingCount": 1100
#     },
#     {
#         "name": "Converse",
#         "description": "Tienda de zapatos y ropa deportiva que ofrece productos de alta calidad para atletas y amantes de la moda.",
#         "category": "Zapatos",
#         "ratingValue": 4.5,
#         "ratingCount": 900
#     }
# ]

# for product in products:
#   text = ""
#   for name, value in product.items():
#     text += f"{str(value).lower()} "
        
#   product['text'] = text
# Measure the speed
start_time = time.time()

# df = LanceDB("products").search("comida", to_vector=True) # vectoriza y busca las similitudes util para errores de gramatica
df = LanceDB("products").search("comida") # busqueda semantica util para velocidad

print(df)

end_time = time.time()
elapsed_time = end_time - start_time
print("Elapsed time for 1000 searches: ", elapsed_time)