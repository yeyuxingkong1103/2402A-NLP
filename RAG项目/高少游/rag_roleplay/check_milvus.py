from pymilvus import MilvusClient

client = MilvusClient(uri='http://localhost:19530')
cols = client.list_collections()
print('Collections:', cols)
for c in cols:
    count = client.get_collection_stats(c).get("row_count", 0)
    print(f'\n  {c}: {count} entities')
    if count > 0:
        try:
            rows = client.query(
                collection_name=c,
                filter='',
                output_fields=['pk','text','source'] if c == 'rag_roleplay' else ['pk','text'],
                limit=3
            )
            for r in rows:
                pk = str(r.get('pk','?'))[:16]
                source = r.get('source','N/A')
                text = str(r.get('text',''))[:80]
                print(f'    pk={pk}... source={source} text={text}...')
        except Exception as e:
            print(f'    Query error: {e}')
