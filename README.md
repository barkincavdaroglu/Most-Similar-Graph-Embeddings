# Most-Similar-Graph-Embeddings

Explanation of this program can be found [here]()

In the main directory, run

```
flask run
```

Then,

```curl
curl --location --request POST '{YOUR_PORT}/get_next_state' \
--header 'Content-Type: application/json' \
--data-raw '{
    "snapshot": "4\nO: a 0\nO: b 1\nO: c 2\nO: d 3\nE: a b 0.51\nE: a c 0.39\nE: b d 0.6\nE: b c 0.4"
}'
```

where {YOUR_PORT} is the URL that app runs on. Using the test data, the output of this call will be a JSON with key "snapshot" and value:

```
4
O: b 0
O: a 1
O: c 2
O: d 3
E: a b 0.58
E: a c 0.29
E: b d 0.58
E: b c 0.19
```

This graph is the next snapshot of the graph that is most similar to our input graph.
