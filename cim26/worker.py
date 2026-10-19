"""One isolated worker per model/round; called only by the sequential runner."""
import json
import sys
import time
import threading
import traceback
import gc
import os
from dataclasses import asdict
from pathlib import Path
import numpy as np
import psutil
from .schema import ModuleSpec, EffectiveSDM, dump_json
from .physics import predict


def main(job_path):
    job=json.loads(Path(job_path).read_text())
    result_path=Path(job["result"])
    try:
        spec=ModuleSpec(**job["spec"])
        params=EffectiveSDM(**job["parameters"])
        inputs=np.load(job["inputs"],allow_pickle=False)
        arrays=[inputs[k] for k in ["g","tm","tc"]]
        def call():
            return predict(job["model"],*arrays,spec,params,tol=job["tolerance"],maxiter=job["max_iterations"])
        warm_start=time.perf_counter_ns()
        first=call()
        warm_s=(time.perf_counter_ns()-warm_start)/1e9
        repeat=job.get("inner_repeats") or max(1,min(10000,int(np.ceil(job["target_batch_s"]/max(warm_s,1e-9)))))
        gc.collect()
        # Some container runtimes expose a host /proc namespace but a nested PID.
        # /proc/self always targets this process, unlike psutil.Process(getpid()).
        if sys.platform.startswith("linux") and Path("/proc/self/statm").exists():
            def resident_bytes():
                return int(Path("/proc/self/statm").read_text().split()[1])*os.sysconf("SC_PAGE_SIZE")
            rss_backend="proc_self_statm"
        else:
            process=psutil.Process()
            def resident_bytes():return process.memory_info().rss
            rss_backend="psutil_process_rss"
        baseline=resident_bytes()
        peak=[baseline];samples=[0];stop=threading.Event()
        def sample():
            while not stop.wait(job["rss_interval_s"]):
                peak[0]=max(peak[0],resident_bytes());samples[0]+=1
        thread=threading.Thread(target=sample,daemon=True)
        thread.start()
        start=time.perf_counter_ns()
        for _ in range(repeat):
            last=call()
        end=time.perf_counter_ns()
        peak[0]=max(peak[0],resident_bytes())
        stop.set();thread.join()
        compute_s=(end-start)/1e9/repeat
        deterministic=bool(np.allclose(first.power_W,last.power_W,rtol=1e-12,atol=1e-9,equal_nan=True))
        # I/O and diagnostics are outside the measured region.
        np.savez_compressed(job["output"],power=last.power_W,converged=last.converged,
                            iterations=last.iterations,status=last.status)
        result=dict(compute_s=compute_s,compute_batch_s=(end-start)/1e9,
            us_per_sample=compute_s/len(arrays[0])*1e6,inner_repeats=repeat,warmup_s=warm_s,
            n_samples=len(arrays[0]),start_ns=start,end_ns=end,pid=os.getpid(),rss_backend=rss_backend,
            rss_baseline_MiB=baseline/2**20,rss_peak_MiB=peak[0]/2**20,
            rss_delta_MiB=(peak[0]-baseline)/2**20,rss_samples=samples[0],
            rss_interval_s=job["rss_interval_s"],rss_peak_kind="sampled_RSS_not_true_OS_high_water_mark",
            nonconverged=int((~last.converged).sum()),valid_predictions=int(np.isfinite(last.power_W).sum()),
            solver_iterations_max=int(last.iterations.max(initial=0)),deterministic=deterministic,
            error="",evaluation_mode="batch_vectorized",exit_code=0)
        dump_json(result_path,result)
        return 0
    except Exception as exc:
        dump_json(result_path,dict(error=f"{type(exc).__name__}: {exc}",traceback=traceback.format_exc(),exit_code=1))
        return 1


if __name__=="__main__":
    sys.exit(main(sys.argv[1]))
