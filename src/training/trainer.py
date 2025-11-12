from src.data.data_loader_collection import DataLoaderCollection
from src.models.base_model import BaseModel
from src.validation.validator import Validator
from src.config.config import GlobalConfig
from src.training.early_stopping import create_early_stopper
from src.training.experiment_logger import create_logger
from src.helpers.profiling import write_cprofile_outputs
import cProfile
from pathlib import Path
from torch.optim import Adam, SGD, RMSprop
import torch
from typing import Any, Dict, List, Optional
import os
import time
import subprocess
from contextlib import suppress
try:
    from torch.profiler import profile, ProfilerActivity, schedule as profiler_schedule
except Exception:
    profile = None
    ProfilerActivity = None
    profiler_schedule = None


class Trainer:
    
    def __init__(self,
                 data_loader: DataLoaderCollection,
                 model: BaseModel,
                 config: GlobalConfig,
                 validator: Validator):
        self.data_loader = data_loader
        self.model = model
        self.global_config = config
        self.config = self.global_config.trainer
        self.losses: dict[int, float] = {}
        self.regularization_losses: dict[int, float] = {}
        self.current_epoch = 0
        self.validator = validator
        self.early_stopping = create_early_stopper(self.config, self.global_config.verbose)
        # tracking the device of the most recent training batch (default CPU)
        try:
            self.last_batch_device = torch.device("cpu")
        except Exception:
            self.last_batch_device = "cpu"
    
    def __init_optimizer(self):
        if self.config.optimizer == "adam":
            self.optimizer = Adam(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "sgd":
            self.optimizer = SGD(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "rmsprop":
            self.optimizer = RMSprop(self.model.parameters(), lr=self.config.learning_rate)
        elif self.config.optimizer == "adamw":
            self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)
        else:
            raise ValueError(f"Unsupported optimizer: {self.config.optimizer}")
    
    def __init_scheduler(self):
        if not self.config.scheduler.enabled:
            self.scheduler = None
            return
        elif self.config.scheduler.type == "step":
            self.scheduler = torch.optim.lr_scheduler.StepLR(self.optimizer, step_size=self.config.scheduler.step_size, gamma=self.config.scheduler.gamma)
        elif self.config.scheduler.type == "exponential":
            self.scheduler = torch.optim.lr_scheduler.ExponentialLR(self.optimizer, gamma=self.config.scheduler.gamma)
        else:
            raise ValueError(f"Unsupported scheduler: {self.config.scheduler}")
    
    def __init_training(self):
        self.__init_optimizer()
        self.__init_scheduler()
        self.epoch_losses: list[float] = []
        self.epoch_regularization_losses: list[float] = []
        self.losses: dict[int, float] = {}
        self.regularization_losses: dict[int, float] = {}
        self.model.prepare_for_training()
        self.current_epoch = 0
        # speed-up for fixed shapes
        try:
            if torch.backends.cudnn.is_available():
                torch.backends.cudnn.benchmark = True
        except Exception:
            pass
        # optional per-batch timing via env flag
        self._timing_enabled = os.environ.get("SPA_TRAIN_TIMING", "0") in ("1", "true", "True")
        if self._timing_enabled:
            self._timing_acc = {
                "batch_time": 0.0,
                "forward": 0.0,
                "backward": 0.0,
                "opt": 0.0,
                "batches": 0,
            }
        # optional memcpy profiling using torch.profiler (short window to limit overhead)
        self._prof_transfers = os.environ.get("SPA_PROFILER_TRANSFERS", "0") in ("1", "true", "True") and profile is not None
        self._prof_steps = int(os.environ.get("SPA_PROFILER_STEPS", "5")) if self._prof_transfers else 0
        self._profiler = None
        self._memcpy_stats: Dict[str, Any] = {}

    def _collect_system_metrics(self) -> Dict[str, Any]:
        """Collect a small snapshot of system metrics (CPU, RAM, GPU) and the last data device.

        This function tries a few strategies and falls back gracefully if optional
        dependencies (psutil/pynvml) are not available.
        """
        metrics: Dict[str, Any] = {}

        # CPU and RAM (prefer psutil)
        try:
            import psutil  # type: ignore
            metrics["cpu_percent"] = psutil.cpu_percent(interval=0.1)
            metrics["mem_percent"] = psutil.virtual_memory().percent
        except Exception:
            # fallback: approximate CPU% from loadavg and /proc/meminfo
            try:
                load1 = os.getloadavg()[0]
                cpu_count = os.cpu_count() or 1
                metrics["cpu_percent"] = round((load1 / cpu_count) * 100.0, 2)
            except Exception:
                metrics["cpu_percent"] = None
            try:
                with open("/proc/meminfo", "r") as f:
                    data = f.read()
                import re as _re
                m_total = int(_re.search(r"MemTotal:\s+(\d+)", data).group(1))
                m_avail = int(_re.search(r"MemAvailable:\s+(\d+)", data).group(1))
                metrics["mem_percent"] = round(100.0 * (m_total - m_avail) / m_total, 2)
            except Exception:
                metrics["mem_percent"] = None

        # GPU: try pynvml, then nvidia-smi, then torch (memory only)
        gpus: List[Dict[str, Optional[Any]]] = []
        try:
            import pynvml  # type: ignore
            pynvml.nvmlInit()
            dev_count = pynvml.nvmlDeviceGetCount()
            for i in range(dev_count):
                handle = pynvml.nvmlDeviceGetHandleByIndex(i)
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                gpus.append({
                    "index": i,
                    "utilization": int(util.gpu),
                    "memory_total_mb": int(meminfo.total // (1024 * 1024)),
                    "memory_used_mb": int(meminfo.used // (1024 * 1024)),
                })
            pynvml.nvmlShutdown()
        except Exception:
            # try nvidia-smi
            try:
                out = subprocess.check_output([
                    "nvidia-smi",
                    "--query-gpu=utilization.gpu,memory.total,memory.used",
                    "--format=csv,noheader,nounits",
                ], encoding="utf-8")
                for idx, line in enumerate(out.strip().splitlines()):
                    parts = [p.strip() for p in line.split(",")]
                    if len(parts) >= 3:
                        util = int(parts[0].replace("%", "")) if parts[0] else None
                        total = int(parts[1])
                        used = int(parts[2])
                        gpus.append({
                            "index": idx,
                            "utilization": util,
                            "memory_total_mb": total,
                            "memory_used_mb": used,
                        })
            except Exception:
                # fallback: torch memory only
                try:
                    if torch.cuda.is_available():
                        for i in range(torch.cuda.device_count()):
                            try:
                                torch.cuda.synchronize(i)
                            except Exception:
                                pass
                            mem_alloc = torch.cuda.memory_allocated(i) // (1024 * 1024)
                            mem_res = torch.cuda.memory_reserved(i) // (1024 * 1024)
                            gpus.append({
                                "index": i,
                                "utilization": None,
                                "memory_allocated_mb": int(mem_alloc),
                                "memory_reserved_mb": int(mem_res),
                            })
                except Exception:
                    pass

        metrics["gpus"] = gpus
        # last observed batch device
        metrics["data_device"] = str(self.last_batch_device)

        # include host->device transfer stats if the data loader tracked them
        try:
            dl = getattr(self, "data_loader", None)
            if dl is not None and hasattr(dl, "transfer_count"):
                metrics["transfer_stats"] = {
                    "transfer_count": int(getattr(dl, "transfer_count", 0)),
                    "transfer_bytes": int(getattr(dl, "transfer_bytes", 0)),
                    "preload_enabled": bool(getattr(dl, "_preload_to_device", False)),
                    "device_preloaded": bool(getattr(dl, "_x_device", None) is not None and getattr(dl, "_y_device", None) is not None),
                }
            else:
                metrics["transfer_stats"] = {"transfer_count": None, "transfer_bytes": None, "preload_enabled": None, "device_preloaded": None}
        except Exception:
            metrics["transfer_stats"] = {"transfer_count": None, "transfer_bytes": None, "preload_enabled": None, "device_preloaded": None}

        return metrics
    
    def train(self):
        if self.global_config.verbose:
            self.__print_training_start()
        logger = create_logger(self.global_config)
        logger.start(
            run_name=self.global_config.run_name,
            config=self.global_config.model_dump(),
        )
        self.__init_training()
        for epoch in range(self.config.epochs):
            self.current_epoch = epoch
            # start profiler at beginning of epoch if enabled
            if self._prof_transfers and self._profiler is None and self._prof_steps > 0:
                try:
                    acts = [ProfilerActivity.CPU]
                    if torch.cuda.is_available():
                        acts.append(ProfilerActivity.CUDA)
                    self._profiler = profile(
                        activities=acts,
                        schedule=profiler_schedule(wait=0, warmup=0, active=self._prof_steps, repeat=1),
                        record_shapes=False,
                        profile_memory=True,
                        with_stack=False,
                    )
                    self._profiler.__enter__()
                except Exception:
                    self._profiler = None
            for x, _ in self.data_loader:
                # track device of the incoming batch so we can report it during validation
                try:
                    if isinstance(x, torch.Tensor):
                        self.last_batch_device = x.device
                    else:
                        # x could be a tuple/dict/numpy; default to cpu
                        self.last_batch_device = getattr(x, "device", torch.device("cpu"))
                except Exception:
                    self.last_batch_device = torch.device("cpu")

                # optional timing start
                if self._timing_enabled:
                    t_batch_start = time.perf_counter()

                self.optimizer.zero_grad()
                # forward (timed)
                if self._timing_enabled:
                    if x.is_cuda:
                        torch.cuda.synchronize()
                    t0 = time.perf_counter()
                loss = self.model.forward(x)
                if self._timing_enabled:
                    if x.is_cuda:
                        torch.cuda.synchronize()
                    self._timing_acc["forward"] += (time.perf_counter() - t0)
                reg_loss = self.model.regularization_loss()
                loss = loss + reg_loss
                # backward (timed)
                if self._timing_enabled:
                    if x.is_cuda:
                        torch.cuda.synchronize()
                    t1 = time.perf_counter()
                loss.backward()
                if self._timing_enabled:
                    if x.is_cuda:
                        torch.cuda.synchronize()
                    self._timing_acc["backward"] += (time.perf_counter() - t1)
                if self.config.grad_clip is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
                # optimizer step (timed)
                if self._timing_enabled:
                    t2 = time.perf_counter()
                self.optimizer.step()
                if self._timing_enabled:
                    if x.is_cuda:
                        torch.cuda.synchronize()
                    self._timing_acc["opt"] += (time.perf_counter() - t2)
                # advance profiler schedule if enabled
                if self._profiler is not None:
                    with suppress(Exception):
                        self._profiler.step()
                self.epoch_losses.append(loss.item())
                self.epoch_regularization_losses.append(reg_loss.item())
                if self._timing_enabled:
                    self._timing_acc["batch_time"] += (time.perf_counter() - t_batch_start)
                    self._timing_acc["batches"] += 1
            if self.scheduler:
                    self.scheduler.step()
            self.losses[epoch] = sum(self.epoch_losses) / len(self.epoch_losses)
            self.regularization_losses[epoch] = sum(self.epoch_regularization_losses) / len(self.epoch_regularization_losses)
            if self.config.validate_per_epoch > 0 and (epoch + 1) % self.config.validate_per_epoch == 0:
                # collect system metrics at validation time and attach them to the
                # validator's per-epoch validations dict so the logger will include them.
                # if we were profiling transfers, finalize and summarize memcpy events now
                if self._profiler is not None:
                    try:
                        events = self._profiler.events()
                        memcpy_count = 0
                        memcpy_cuda_time_us = 0
                        for ev in events:
                            name = str(getattr(ev, "name", "")).lower()
                            if "memcpy" in name:
                                memcpy_count += 1
                                # prefer CUDA time if available
                                memcpy_cuda_time_us += int(getattr(ev, "cuda_time_total", 0))
                        self._memcpy_stats = {
                            "memcpy_events": memcpy_count,
                            "memcpy_cuda_time_us": memcpy_cuda_time_us,
                            "window_batches": self._prof_steps,
                        }
                    except Exception:
                        self._memcpy_stats = {}
                    finally:
                        # close profiler and clear until next epoch/window
                        with suppress(Exception):
                            self._profiler.__exit__(None, None, None)
                        self._profiler = None

                system_metrics = self._collect_system_metrics()
                print(f"System Metrics at Epoch {epoch + 1}: {system_metrics}", flush=True)
                # run validation (may update validator.validations[epoch])
                self.validator.validate_epoch(epoch, self.optimizer)
                # attach system metrics into the validation output for this epoch
                val = self.validator.validations.get(epoch, {})
                val["system_metrics"] = system_metrics
                if self._memcpy_stats:
                    val.setdefault("system_metrics", {})["memcpy_profiler"] = self._memcpy_stats
                # attach timing averages if enabled
                if self._timing_enabled and self._timing_acc["batches"] > 0:
                    b = max(1, self._timing_acc["batches"])
                    val["timings"] = {
                        "batch_time_avg": self._timing_acc["batch_time"] / b,
                        "forward_avg": self._timing_acc["forward"] / b,
                        "backward_avg": self._timing_acc["backward"] / b,
                        "opt_avg": self._timing_acc["opt"] / b,
                        "batches": int(self._timing_acc["batches"]),
                    }
                    # reset accumulator after logging per validation window
                    self._timing_acc.update({"batch_time":0.0,"forward":0.0,"backward":0.0,"opt":0.0,"batches":0})
                self.validator.validations[epoch] = val
            lr = self.optimizer.param_groups[0].get('lr')
            val = self.validator.validations.get(epoch, {})
            logger.log_epoch(
                step=epoch,
                train_total_loss=self.losses[epoch],
                train_reg_loss=self.regularization_losses[epoch],
                lr=lr,
                val_metrics=val,
                epoch=epoch + 1,
                run_seed=int(self.global_config.seed),
            )
            if self.early_stopping is not None and self.early_stopping.step(self.losses[epoch], self.model, epoch, optimizer=self.optimizer):
                break
            self.epoch_losses.clear()
            self.epoch_regularization_losses.clear()
            if self.global_config.verbose:
                self.__print_epoch()
        logger.finish()
    
    def train_profiled(self, out_dir: str | Path, *, basename: str = "train", sort: str = "cumulative") -> None:
        prof = cProfile.Profile()
        prof.enable()
        try:
            self.train()
        finally:
            prof.disable()
            write_cprofile_outputs(prof, out_dir, basename=basename, sort=sort)
    
    def get_losses(self) -> dict[int, float]:
        assert self.losses, "Training has not been run yet."
        return self.losses
    
    def reset(self):
        self.current_epoch = 0
        if self.config.early_stopping:
            self.early_stopping = create_early_stopper(self.config, self.global_config.verbose)
        else:
            self.early_stopping = None
            
    def __print_training_start(self):
        print("\n" + "=" * 60)
        print(f"🚀 Starting Training ({self.config.epochs} Epochs)".center(60, "="))
        print("=" * 60 + "\n")

    def __print_epoch(self):
        epoch_str = f"Epoch {self.current_epoch + 1}/{self.config.epochs}"
        total_loss_str = f"Total Loss: {self.losses[self.current_epoch]:.5f}"
        loss_str = f"Loss: {self.losses[self.current_epoch]-self.regularization_losses[self.current_epoch]:.5f}"
        reg_loss_str = f"Reg Loss: {self.regularization_losses[self.current_epoch]:.5f}" if self.regularization_losses else "Reg Loss: 0.0000"
        
        print(f"{epoch_str:<15} | {total_loss_str} | {loss_str} | {reg_loss_str}", flush=True)

    def __str__(self) -> str:
        return f"Trainer(config={self.config})"