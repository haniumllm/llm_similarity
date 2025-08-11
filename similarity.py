# search_query_cosine.py
import os, json, requests, xmltodict, sys
from urllib.parse import quote_plus
from dotenv import load_dotenv
from openai import OpenAI
import numpy as np
import argparse

load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip()
KIPRIS_API_KEY = os.getenv("KIPRIS_API_KEY", "").strip()
if not OPENAI_API_KEY or not KIPRIS_API_KEY:
    raise SystemExit("환경변수 OPENAI_API_KEY / KIPRIS_API_KEY 필요")

client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = "text-embedding-3-small"
SEARCH_URL = "http://plus.kipris.or.kr/kipo-api/kipi/patUtiModInfoSearchSevice/getWordSearch"
BAD_ABS = {"", None, "내용 없음", "내용 없음."}
 

def make_text(title: str, abstr: str) -> str:
    title = (title or "").strip()
    abstr = (abstr or "").strip()
    return title if abstr in BAD_ABS else f"{title} {abstr}"

def embed_texts(texts):
    resp = client.embeddings.create(model=EMBED_MODEL, input=texts)
    arr = [np.array(d.embedding, dtype=float) for d in resp.data]
    return np.vstack(arr)  # (N, D)

def cosine_matrix(a, b):
    a_n = a / (np.linalg.norm(a, axis=1, keepdims=True) + 1e-12)
    b_n = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-12)
    return a_n @ b_n.T

def score3_from_cos(cos_val: float) -> str:
    sim01 = (cos_val + 1.0) / 2.0
    v = int(round(max(0.0, min(1.0, sim01)) * 100))
    return f"{v:03d}"

# --------- KIPRIS 호출 ---------
def kipris_search(word: str, year=0, patent="Y", utility="Y", rows=10, page=1):
    url = (
        f"{SEARCH_URL}?word={quote_plus(word)}&year={year}"
        f"&patent={patent}&utility={utility}"
        f"&numOfRows={rows}&pageNo={page}&ServiceKey={KIPRIS_API_KEY}"
    )
    r = requests.get(url, timeout=(10, 60))  # connect/read 타임아웃
    r.raise_for_status()
    data = xmltodict.parse(r.content)
    items = data.get("response", {}).get("body", {}).get("items", {}).get("item")
    if not items:
        return []
    return items if isinstance(items, list) else [items]

# --------- 메인 ---------
def main():
    ap = argparse.ArgumentParser(description="KIPRIS 검색 + 임베딩 코사인 유사도")
    ap.add_argument("--search", required=True, help="KIPRIS 검색어(짧게)")
    ap.add_argument("--query",  required=True, help="유사도 계산용 문장(길게 가능)")
    ap.add_argument("--rows", type=int, default=5, help="페이지당 건수 (기본 5)")
    args = ap.parse_args()

    # 1) KIPRIS 검색은 --search 사용
    items = kipris_search(args.search, rows=args.rows, page=1)
    if not items:
        print("검색 결과 없음")
        return

    # 2) 특허 텍스트 준비 (제목+초록)
    docs = []
    for it in items:
        title = it.get("inventionTitle") or it.get("inventionName") or ""
        abstr = it.get("astrtCont") or ""
        docs.append((it, title, abstr, make_text(title, abstr)))

    # 3) 임베딩: 유사도 계산은 --query(긴 문장) vs 문서들
    all_vecs = embed_texts([args.query] + [t for *_, t in docs])
    q_vec, d_mat = all_vecs[0:1, :], all_vecs[1:, :]

    # 4) 코사인 유사도
    cos_scores = cosine_matrix(q_vec, d_mat).ravel()

    # 5) 정렬 + 출력
    ranked = sorted(zip(docs, cos_scores), key=lambda x: x[1], reverse=True)
    for (it, title, abstr, _), cosv in ranked:
        score = score3_from_cos(cosv)
        appno = it.get("applicationNumber", "")
        ipc = it.get("ipcNumber", "")
        reg = it.get("registerStatus", "")
        print(f"{score}  ({cosv:.4f})  {title}  (appNo:{appno})")
        if abstr and abstr not in BAD_ABS:
            print(f"    - 초록: {abstr[:150]}{'…' if len(abstr) > 150 else ''}")
        print(f"    - IPC:{ipc}  상태:{reg}")

    # 6) JSON도 필요하면 주석 해제
    # out = []
    # for (it, title, abstr, _), cosv in ranked:
    #     out.append({
    #         "similarityScore": score3_from_cos(cosv),
    #         "cosine": float(cosv),
    #         "inventionTitle": title,
    #         "astrtCont": abstr,
    #         "applicationNumber": it.get("applicationNumber", ""),
    #         "ipcNumber": it.get("ipcNumber", ""),
    #         "registerStatus": it.get("registerStatus", "")
    #     })
    # print(json.dumps(out, ensure_ascii=False, indent=2))

if __name__ == "__main__":
    main()
