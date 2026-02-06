#!/bin/bash

# --- Configuration ---
TEXT_PORT=8000
IMAGE_PORT=8001
E5_PORT=9100
FAISS_PORT=9200

VECTOR_CONFIG="data/local_KB/vector_collections.json"

RETRIEVER_MODEL="intfloat/e5-base-v2"

# --- Cleanup Function ---
pids=()
cleanup() {
    echo "Caught signal, shutting down services..."
    for pid in "${pids[@]}"; do
        kill -9 -$pid 2>/dev/null
    done
    echo "All services stopped."
}
trap cleanup EXIT SIGINT

# --- Service Launch ---

echo "Starting services..."

# Start embedding service
setsid python -m search_engine.servers.embedding_service \
    --model_name $RETRIEVER_MODEL \
    --device cuda \
    --port $E5_PORT \
    --max_batch_size 256 &
pids+=($!)

# Wait for embedding service health (avoid startup race)
echo "   Waiting for Embedding Service..."
for i in {1..60}; do
    if curl -sf "http://127.0.0.1:${E5_PORT}/health" >/dev/null; then
        echo "   Embedding Service is ready."
        break
    fi
    sleep 1
done

# Start vector search service
setsid python -m search_engine.servers.vector_search_service \
    --config $VECTOR_CONFIG \
    --port $FAISS_PORT &
pids+=($!)

# Start text retrieval server
setsid python -m search_engine.servers.retrieval_server \
    --port $TEXT_PORT \
    --corpus_path "data/local_KB/text_corpus.jsonl" \
    --retriever_name "e5" \
    --retriever_model $RETRIEVER_MODEL \
    --topk 3 \
    --embedding_endpoint http://127.0.0.1:$E5_PORT/embed \
    --vector_endpoint http://127.0.0.1:$FAISS_PORT/search \
    --vector_collection text &
pids+=($!)

# Start image retrieval server
setsid python -m search_engine.servers.image_retrieval_server \
    --port $IMAGE_PORT \
    --corpus_path "data/local_KB/image_corpus.jsonl" \
    --embedding_endpoint http://127.0.0.1:$E5_PORT/embed \
    --vector_endpoint http://127.0.0.1:$FAISS_PORT/search \
    --vector_collection image \
    --batch_size 256 \
    --topk 1 &
pids+=($!)

# --- Keep Alive ---

echo "------------------------------------------------"
echo "All services started successfully!"
echo "Text Retrieval: http://127.0.0.1:$TEXT_PORT"
echo "Image Retrieval: http://127.0.0.1:$IMAGE_PORT"
echo "------------------------------------------------"
echo "Press Ctrl+C to stop all services."

wait -n "${pids[@]}"