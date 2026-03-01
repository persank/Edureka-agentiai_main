# Create a chroma client
# It is ephemeral - starts a Chroma server in memory
# Data will be deleted when the program ends
import chromadb
chroma_client = chromadb.Client()

# Create a collection
collection = chroma_client.create_collection(name="my_collection")
# We can also add a specific distance metric to be used
# collection = chroma_client.create_collection(
#     name="my_collection", 
#     distance_metric="cosine") # default
# Other options for distance: l2 (euclidean distance)
# and ip (inner product)

# Add some text to the collection
# Chroma will embed the text on its own using its own model
collection.add(
    ids=["id1", "id2"],
    documents=[
        "This is a document about pineapple",
        "This is a document about oranges"
    ]
)

# Query the collection
results = collection.query(
    query_texts=["This is a query document about hawaii"], # Chroma will embed this for you
    n_results=2 # how many results to return
)
print(results)




