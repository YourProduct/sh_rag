import os
import glob
import argparse
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from yandex_gpt.interface import YandexGPT
from yandex_gpt.schemas import Messages, SystemMessage, UserMessage


def load_docs(path: str):
    docs = []
    names = []
    for file in glob.glob(os.path.join(path, '*.txt')):
        with open(file, 'r', encoding='utf-8') as f:
            docs.append(f.read())
            names.append(file)
    return docs, names


def build_index(docs):
    vectorizer = TfidfVectorizer()
    embeddings = vectorizer.fit_transform(docs)
    return vectorizer, embeddings


def search(query, vectorizer, embeddings, docs, top_k=3):
    q_emb = vectorizer.transform([query])
    sims = cosine_similarity(q_emb, embeddings).flatten()
    top_indices = sims.argsort()[::-1][:top_k]
    return [docs[i] for i in top_indices]


def generate_answer(question: str, context: str, token: str, folder_id: str):
    gpt = YandexGPT(oauth_token=token, folder_id=folder_id)
    messages = Messages(
        SystemMessage("Answer the question based on the context."),
        UserMessage(f"Context: {context}\nQuestion: {question}")
    )
    resp = gpt.completion(messages)
    return resp.alternatives[0].message.text


def answer_question(question, vectorizer, embeddings, docs, token, folder_id, top_k=3):
    retrieved = search(question, vectorizer, embeddings, docs, top_k=top_k)
    context = "\n".join(retrieved)
    return generate_answer(question, context, token, folder_id)


def main():
    parser = argparse.ArgumentParser(description="Simple RAG pipeline")
    parser.add_argument('question', help='Question to ask')
    parser.add_argument('--docs', default='docs', help='Path to documents')
    parser.add_argument('--top_k', type=int, default=3, help='Number of docs to retrieve')
    parser.add_argument('--token', default=os.environ.get('YANDEX_TOKEN'), help='Yandex OAuth token')
    parser.add_argument('--folder', default=os.environ.get('YANDEX_FOLDER_ID'), help='Yandex folder id')
    args = parser.parse_args()

    if not args.token or not args.folder:
        parser.error('Yandex credentials must be provided via arguments or environment variables')

    docs, _ = load_docs(args.docs)
    if not docs:
        print('No documents found in', args.docs)
        return

    vectorizer, embeddings = build_index(docs)
    answer = answer_question(args.question, vectorizer, embeddings, docs, args.token, args.folder, args.top_k)
    print('Answer:', answer)


if __name__ == '__main__':
    main()
