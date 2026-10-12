import time
import threading
import requests

BASE_URL = 'http://localhost:9380'
CHAT_ID = '8fbaaf18c57611f180a2c6a1cd97b544'
API_KEY = 'ragflow-你的完整KEY'
QUESTION = 'CN1037529C的发明人是谁'
CONCURRENCY = 20       # 并发数（优化前后都用20）
DURATION = 60          # 压测时长（秒）

results = []
lock = threading.Lock()


def worker(end_time):
    while time.time() < end_time:
        start = time.time()
        try:
            r = requests.post(
                f'{BASE_URL}/api/v1/chats_openai/{CHAT_ID}/chat/completions',
                headers={'Authorization': f'Bearer {API_KEY}',
                         'Content-Type': 'application/json'},
                json={'model': 'glm-4-flash',
                      'messages': [{'role': 'user', 'content': QUESTION}],
                      'stream': False},
                timeout=60
            )
            elapsed = time.time() - start
            with lock:
                results.append({'elapsed': elapsed, 'status': r.status_code})
        except Exception as e:
            with lock:
                results.append({'elapsed': time.time() - start,
                                'status': 'error', 'error': str(e)})


end_time = time.time() + DURATION
threads = [threading.Thread(target=worker, args=(end_time,))
           for _ in range(CONCURRENCY)]
for t in threads:
    t.start()
for t in threads:
    t.join()

elapsed = sorted([r['elapsed'] for r in results])
success = sum(1 for r in results if r.get('status') == 200)
fail = len(results) - success

print(f'Total: {len(results)}, Success: {success}, Fail: {fail}')
print(f'Throughput: {len(results) / DURATION:.2f} req/s')
print(f'Avg: {sum(elapsed) / len(elapsed):.3f}s')
print(f'P50: {elapsed[len(elapsed) // 2]:.3f}s')
print(f'P95: {elapsed[int(len(elapsed) * 0.95)]:.3f}s')
print(f'P99: {elapsed[int(len(elapsed) * 0.99)]:.3f}s')
print(f'Max: {max(elapsed):.3f}s')