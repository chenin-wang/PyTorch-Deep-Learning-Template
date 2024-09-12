import torch
import torchvision.models as models
from torch.profiler import profile, record_function, ProfilerActivity
from torch.profiler import schedule
from accelerate import Accelerator, ProfileKwargs


class ModelProfiler:
    def __init__(self, model, inputs):
        self.model = model
        self.inputs = inputs

    def analyze_execution_time(self):
        with profile(activities=[ProfilerActivity.CPU], record_shapes=True) as prof:
            self.model(self.inputs)
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))

    def analyze_memory_consumption(self):
        with profile(activities=[ProfilerActivity.CPU], profile_memory=True, record_shapes=True) as prof:
            self.model(self.inputs)
        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    def export_chrome_trace(self, filename="trace.json"):
        """
        examine the sequence of profiled operators and CUDA kernels in Chrome trace viewer (chrome://tracing)
        """
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            self.model(self.inputs)
        prof.export_chrome_trace(filename)

    def analyze_long_running_jobs(self):
        my_schedule = schedule(
            skip_first=10,
            wait=5,
            warmup=1,
            active=3,
            repeat=2
        )

        def trace_handler(p):
            output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
            print(output)
            p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            schedule=my_schedule,
            on_trace_ready=trace_handler
        ) as p:
            for idx in range(8):
                self.model(self.inputs)
                p.step()

    def estimate_flops(self):
        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            with_flops=True
        ) as prof:
            self.model(self.inputs)
        print(prof.key_averages().table(sort_by="flops", row_limit=10))


class ModelAcceleratorProfiler:
    def __init__(self, model, inputs):
        self.model = model
        self.inputs = inputs

    def analyze_execution_time(self):
        profile_kwargs = ProfileKwargs(
            activities=["cpu"],
            record_shapes=True
        )
        accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])
        model = accelerator.prepare(self.model)

        with accelerator.profile() as prof:
            with torch.no_grad():
                model(self.inputs)

        print(prof.key_averages().table(sort_by="cpu_time_total", row_limit=10))
        print(prof.key_averages(group_by_input_shape=True).table(sort_by="cpu_time_total", row_limit=10))

    def analyze_memory_consumption(self):
        profile_kwargs = ProfileKwargs(
            activities=["cpu"],
            profile_memory=True,
            record_shapes=True
        )
        accelerator = Accelerator(cpu=True, kwargs_handlers=[profile_kwargs])
        model = accelerator.prepare(self.model)

        with accelerator.profile() as prof:
            model(self.inputs)

        print(prof.key_averages().table(sort_by="self_cpu_memory_usage", row_limit=10))

    def export_chrome_trace(self, output_dir="trace"):
        profile_kwargs = ProfileKwargs(
            activities=["cpu", "cuda"],
            output_trace_dir=output_dir
        )
        accelerator = Accelerator(kwargs_handlers=[profile_kwargs])
        model = accelerator.prepare(self.model)

        with accelerator.profile() as prof:
            model(self.inputs)

    def analyze_long_running_jobs(self):
        def trace_handler(p):
            output = p.key_averages().table(sort_by="self_cuda_time_total", row_limit=10)
            print(output)
            p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

        profile_kwargs = ProfileKwargs(
            activities=["cpu", "cuda"],
            schedule_option={"wait": 5, "warmup": 1, "active": 3, "repeat": 2, "skip_first": 10},
            on_trace_ready=trace_handler
        )
        accelerator = Accelerator(kwargs_handlers=[profile_kwargs])
        model = accelerator.prepare(self.model)

        with accelerator.profile() as prof:
            for idx in range(8):
                model(self.inputs)
                prof.step()

    def estimate_flops(self):
        profile_kwargs = ProfileKwargs(
            with_flops=True
        )
        accelerator = Accelerator(kwargs_handlers=[profile_kwargs])

        with accelerator.profile() as prof:
            self.model(self.inputs)

        print(prof.key_averages().table(sort_by="flops", row_limit=10))


if __name__ == "__main__":
    model = models.resnet18()
    inputs = torch.randn(5, 3, 224, 224)
    
    profiler = ModelProfiler(model, inputs)
    
    profiler.analyze_execution_time()
    profiler.analyze_memory_consumption()
    profiler.export_chrome_trace()
    profiler.analyze_long_running_jobs()
    profiler.estimate_flops()

    accelerator_profiler = ModelAcceleratorProfiler(model, inputs)

    accelerator_profiler.analyze_execution_time()
    accelerator_profiler.analyze_memory_consumption()
    accelerator_profiler.export_chrome_trace()
    accelerator_profiler.analyze_long_running_jobs()
    accelerator_profiler.estimate_flops()