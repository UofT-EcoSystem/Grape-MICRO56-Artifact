import urllib.request

from ..module import replace

tvm_version = "0.12.dev0"

popen_pool_url = f"https://raw.githubusercontent.com/apache/tvm/v{tvm_version}/python/tvm/contrib/popen_pool.py"
popen_pool = urllib.request.urlopen(popen_pool_url)
with open("quik_fix/popen/pool.py", "wt", encoding="utf-8") as popen_pool_fout:
    popen_pool_str = replace("tvm.exec.popen_worker", "quik_fix.popen.worker")(
        popen_pool.read().decode()
    )
    popen_pool_fout.write(popen_pool_str)


popen_worker_url = f"https://raw.githubusercontent.com/apache/tvm/v{tvm_version}/python/tvm/exec/popen_worker.py"
popen_worker = urllib.request.urlopen(popen_worker_url)
with open("quik_fix/popen/worker.py", "wt", encoding="utf-8") as popen_worker_fout:
    popen_worker_str = replace("tvm.contrib.popen_pool", ".pool")(
        popen_worker.read().decode()
    )
    popen_worker_fout.write(popen_worker_str)
