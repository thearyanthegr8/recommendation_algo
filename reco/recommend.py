import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import PIL


def get_similar(embedding, k):
    model_similar_items = NearestNeighbors(n_neighbors=k, algorithm="ball_tree").fit(
        embedding
    )
    distances, indices = model_similar_items.kneighbors(embedding)

    return distances, indices


def show_similar(item_index, item_similar_indices, item_encoder):
    s = item_similar_indices[item_index]
    movie_ids = item_encoder.inverse_transform(s)

    images = []
    for movie_id in movie_ids:
        img_path = "data/posters/" + str(movie_id) + ".0.jpg"
        try:
            images.append(mpimg.imread(img_path))
        except FileNotFoundError:
            # Try alternative path without the ".0"
            try:
                img_path = "data/posters/" + str(movie_id) + ".jpg"
                images.append(mpimg.imread(img_path))
            except FileNotFoundError:
                print(f"Image not found for movie ID {movie_id}")
                continue

    if not images:
        print("No images found to display")
        return

    plt.figure(figsize=(20, 10))
    columns = 5
    rows = (len(images) // columns) + (1 if len(images) % columns > 0 else 0)

    for i, image in enumerate(images):
        plt.subplot(rows, columns, i + 1)
        plt.axis("off")
        plt.imshow(image)

    plt.tight_layout()  # Adjust the layout
    plt.show()
