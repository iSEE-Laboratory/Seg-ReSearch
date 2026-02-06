import json
import os
import numpy as np
import faiss
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm

CORPUS_PATH = "data/local_KB/image_corpus.jsonl"
INDEX_OUTPUT_PATH = "data/local_KB/image_e5_Flat.index"
MODEL_NAME = "intfloat/e5-base-v2"


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def main():
    print(f"1. Loading model: {MODEL_NAME} ...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModel.from_pretrained(MODEL_NAME).cuda()
    model.eval()

    print(f"2. Reading image corpus from {CORPUS_PATH} ...")
    texts = []

    # 读取 image_corpus.jsonl
    with open(CORPUS_PATH, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            item = json.loads(line)
            image_name = item.get('name', '')
            texts.append(f"passage: {image_name}")

    print(f"   Total {len(texts)} images found.")

    print("3. Encoding texts to vectors ...")
    batch_size = 64
    all_embeddings = []

    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i: i + batch_size]
        encoded_input = tokenizer(batch_texts, padding=True, truncation=True, max_length=512, return_tensors='pt').to(
            'cuda')

        with torch.no_grad():
            model_output = model(**encoded_input)
            sentence_embeddings = mean_pooling(model_output, encoded_input['attention_mask'])
            sentence_embeddings = torch.nn.functional.normalize(sentence_embeddings, p=2, dim=1)
            all_embeddings.append(sentence_embeddings.cpu().numpy())

    all_embeddings = np.concatenate(all_embeddings, axis=0)
    print(f"   Embeddings shape: {all_embeddings.shape}")

    print("4. Building FAISS index ...")
    d = all_embeddings.shape[1]
    index = faiss.IndexFlatIP(d)
    index.add(all_embeddings)

    os.makedirs(os.path.dirname(INDEX_OUTPUT_PATH), exist_ok=True)

    print(f"5. Saving index to {INDEX_OUTPUT_PATH} ...")
    faiss.write_index(index, INDEX_OUTPUT_PATH)
    print("Done! Index build complete.")


if __name__ == "__main__":
    main()