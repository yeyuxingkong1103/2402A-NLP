from __future__ import annotations
import argparse,json,urllib.request,urllib.error
def main():
 p=argparse.ArgumentParser();p.add_argument("--base-url",default="http://localhost:9380");a=p.parse_args();url=a.base_url.rstrip("/")+"/v1/system/metrics";r={"url":url,"reachable":False,"status":None,"error":None}
 try:
  with urllib.request.urlopen(url,timeout=3) as x:r.update(reachable=True,status=x.status)
 except Exception as exc:r["error"]=f"{type(exc).__name__}: {exc}"
 print(json.dumps(r,ensure_ascii=False,indent=2))
if __name__=="__main__":main()
