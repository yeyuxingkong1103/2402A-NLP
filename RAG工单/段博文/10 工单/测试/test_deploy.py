# -*- coding: utf-8 -*-
# 工单编号：人工智能NLP-RAG-金融问答系统部署
"""
Docker 容器部署验收测试脚本

测试内容：
1. 容器启动与运行
2. 容器数据管理（持久化）
3. 网络配置
4. 金融问答服务测试
"""

import requests
import time
import json

BASE = "http://127.0.0.1:8000"

# 10 道金融问答测试题
QUESTIONS = [
    ("平安银行2019年的董事长是谁？", ["谢永林"]),
    ("招商银行2019年的净利润是多少亿元？", ["928"]),
    ("邮储银行2019年的三大风险是什么？", ["信用风险", "市场风险"]),
    ("中信证券2020年的营业收入是多少？", ["543"]),
    ("中国人寿2020年的保费收入是多少？", ["6122"]),
    ("国泰君安2021年的法定代表人是谁？", ["贺青"]),
    ("中国太保2021年的数字化转型战略是什么？", ["数字化"]),
    ("招商证券2021年的总资产是多少？", ["5597"]),
    ("中国平安2019年的保险资金投资组合规模是多少？", ["3.21"]),
    ("平安银行和招商银行哪家净利润更高？", ["招商", "928"]),
]


def test_container_health():
    """测试 1：容器健康检查"""
    print("\n" + "=" * 60)
    print("测试 1：容器启动与运行")
    print("=" * 60)

    try:
        r = requests.get(f"{BASE}/api/health", timeout=10)
        assert r.status_code == 200, "健康检查接口返回非 200"
        data = r.json()
        assert data["status"] == "ok", "服务状态异常"
        print(f"  ✅ 健康检查通过")
        print(f"     集合块数: {data['collection_count']}")
        print(f"     版本: {data.get('version', 'N/A')}")
        return True
    except Exception as e:
        print(f"  ❌ 健康检查失败: {e}")
        return False


def test_data_persistence():
    """测试 2：容器数据管理"""
    print("\n" + "=" * 60)
    print("测试 2：容器数据管理")
    print("=" * 60)

    try:
        r = requests.get(f"{BASE}/api/health", timeout=10)
        data = r.json()
        count = data["collection_count"]
        assert count > 0, "集合为空，数据未持久化"
        print(f"  ✅ 数据持久化正常")
        print(f"     当前集合块数: {count}")
        return True
    except Exception as e:
        print(f"  ❌ 数据管理测试失败: {e}")
        return False


def test_network():
    """测试 3：网络配置"""
    print("\n" + "=" * 60)
    print("测试 3：网络配置")
    print("=" * 60)

    try:
        # 测试检索接口（验证应用与 Milvus 通信）
        r = requests.get(f"{BASE}/api/search", params={
            "query": "测试", "top_k": 1
        }, timeout=30)
        assert r.status_code == 200, "检索接口返回非 200"
        print(f"  ✅ 网络配置正常")
        print(f"     应用与 Milvus 通信正常")
        return True
    except Exception as e:
        print(f"  ❌ 网络测试失败: {e}")
        return False


def test_qa_service():
    """测试 4：金融问答服务"""
    print("\n" + "=" * 60)
    print("测试 4：金融问答服务（10 题）")
    print("=" * 60)

    results = []
    for i, (q, keywords) in enumerate(QUESTIONS, 1):
        try:
            t0 = time.time()
            r = requests.post(f"{BASE}/api/chat", json={
                "query": q, "top_k": 3, "stream": False
            }, timeout=120)
            elapsed = time.time() - t0

            data = r.json()
            answer = data.get("answer", "")
            ok = any(k in answer for k in keywords)

            results.append(ok)
            status = "✓" if ok else "✗"
            print(f"  [{status}] Q{i}: {q[:30]}... ({elapsed:.1f}s)")
            if not ok:
                print(f"       答案: {answer[:100]}...")
        except Exception as e:
            print(f"  [✗] Q{i}: 失败 - {e}")
            results.append(False)

    passed = sum(results)
    total = len(results)
    rate = passed / total * 100
    print(f"\n  准确率: {passed}/{total} = {rate:.1f}%")
    return rate >= 80


def main():
    print("\n" + "#" * 60)
    print("#  RAG 金融问答系统 Docker 部署验收测试")
    print("#" * 60)

    tests = [
        ("容器启动与运行", test_container_health),
        ("容器数据管理", test_data_persistence),
        ("网络配置", test_network),
        ("金融问答服务", test_qa_service),
    ]

    results = {}
    for name, test_func in tests:
        results[name] = test_func()

    print("\n" + "#" * 60)
    print("#  测试结果汇总")
    print("#" * 60)

    all_passed = True
    for name, passed in results.items():
        status = "✅ 通过" if passed else "❌ 失败"
        print(f"  {name}: {status}")
        if not passed:
            all_passed = False

    print(f"\n  总体: {'✅ 全部通过' if all_passed else '❌ 存在失败项'}")

    # 保存结果
    with open("deploy_test_results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
