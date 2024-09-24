import functools
import logging
import os
import socket

from adaptr.src.client.python import host, worker
from adaptr.src.client.python.callbacks.start_stop import stop
from adaptr.src.client.python.callbacks.sync_state import (
    send_ckpt,
    recv_ckpt,
    no_op_ckpt,
)
from main import train_process
from omegaconf import OmegaConf
import adaptr_core


def custom_start_callback(
    host: host.Host,
    command_metadata: adaptr_core.CommandMetadata,
    host_info: adaptr_core.HostInfo,
    peer_host_info: adaptr_core.HostInfo,
):
    """Callback definition for GPU training start.

    Args:
        host (host.Host): Reference to the host.
        command_metadata (adaptr_core.CommandMetadata): Metadata for the host's comamnd.
        host_info (adaptr_core.HostInfo): New host information for the host.
        peer_host_info (adaptr_core.WorkerInfo): New host information on a peer. This argument is not used for this callback.
    """
    logging.info(f"{host.host_name}: Executing start callback")

    for worker_info in host_info.workers:
        worker_name = worker_info.worker_name
        if worker_name in host.workers:
            worker = host.workers[worker_name]

            # Update worker attributes
            worker.update_info(worker_info=worker_info)

            # Update config
            cfg = OmegaConf.load("gpt2_train_cfg.yaml")
            # Adjust number of data replicas and data replica size to match environment
            if os.environ.get("NUM_DATA_REPLICAS", None) is not None:
                cfg.trainer_config.num_data_replicas = int(
                    os.environ.get("NUM_DATA_REPLICAS")
                )
            if os.environ.get("DATA_REPLICA_SIZE", None) is not None:
                cfg.trainer_config.data_replica_size = int(
                    os.environ.get("DATA_REPLICA_SIZE")
                )
            old_num_data_replicas = cfg.trainer_config.num_data_replicas
            cfg.trainer_config.num_data_replicas = (
                command_metadata.new_world_size // cfg.trainer_config.data_replica_size
            )
            cfg.trainer_config.global_batch_size *= (
                old_num_data_replicas // cfg.trainer_config.num_data_replicas
            )
            cfg.trainer_config.global_batch_size = int(
                cfg.trainer_config.global_batch_size
            )

            # start training loop
            kwargs = {
                "global_rank": worker.global_rank,
                "world_size": command_metadata.new_world_size,
                "cfg": cfg,
            }
            logging.info(
                f"{host.host_name}: Starting training for {worker.worker_name}, kwargs: {kwargs}"
            )
            worker.start_workload(func=worker.workload_func, kwargs=kwargs)
        else:
            logging.warning(
                f"{host.host_name}: could not find python process for {worker_name}."
            )


def start_worker(
    host_instance: host.Host,
    local_rank: int,
    world_size,
    port: int,
    redis_port: int,
):
    global_rank = local_rank + (host_instance.host_rank * 8)

    # Initialize worker
    w = worker.Worker(port, host_instance.host_address, local_rank, global_rank)
    host_instance.register_workers([w])

    # Initialize worker workload functions
    w.workload_func = functools.partial(
        train_process, redis_port=redis_port, host_name=host_instance.host_name
    )


def main():
    # Set up logging (optional, but recommended)
    logging.basicConfig(level=logging.INFO)

    # Replace with your actual GCP project ID
    project_id = str(os.environ.get("PROJECT", "supercomputer-testing"))

    # Define ports for the host
    supervisor_port = int(os.environ.get("PORT", 60060))
    redis_port = int(os.environ.get("REDIS_PORT", 6379))

    # Define host_rank (usually 0 for a single host)
    host_rank = int(os.environ.get("HOST_RANK", 0))
    # Define world_size
    world_size = int(os.environ["WORLD_SIZE"])

    # Define torch address and port
    if os.environ.get("MASTER_ADDR") is None:
        os.environ["MASTER_ADDR"] = "localhost"
    if os.environ.get("MASTER_PORT") is None:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.bind(("", 0))
        port = sock.getsockname()[1]
        sock.close()
        os.environ["MASTER_PORT"] = str(port)

    # Create a Host instance
    job_name = f"{os.environ['JOB_NAME']}"
    job_index = f"{os.environ['JOB_INDEX']}"
    host_name = f"{job_name}-{job_index}.{os.environ['SERVICE_NAME']}"
    h = host.Host(
        project_id=project_id,
        supervisor_port=supervisor_port,
        redis_port=redis_port,
        host_rank=host_rank,
        network_reachable_hostname=host_name,
    )

    # Register callbacks
    h.register_callback("start", custom_start_callback)
    h.register_callback("stop", stop)
    h.register_callback("send_ckpt", send_ckpt)
    h.register_callback("recv_ckpt", recv_ckpt)
    h.register_callback("no_op_ckpt", no_op_ckpt)

    # Start workers
    for local_rank in range(8):
        worker_port = supervisor_port + local_rank + 1
        start_worker(h, local_rank, world_size, worker_port, redis_port)

    h.start_heartbeat()

    # TODO: Keep the main thread alive to process callbacks (replace with your actual logic)
    while True:
        pass

    # Shutdown the Host when done
    h.shutdown()


if __name__ == "__main__":
    main()
