# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

# import argparse
# import logging
# import subprocess
# from datetime import datetime

# import isaac_ray_util
# import ray

# # Set up logging
# logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


# @ray.remote
# def execute_command(command: str, test_mode: bool = False) -> str:
#     import torch

#     """
#     Execute a specified command or check GPU availability if in test mode.
#     Args:
#         command (str): The command to execute.
#         test_mode (bool): Flag to run GPU test with nvidia-smi.
#     Returns:
#         str: A description of the result or GPUs on the system.
#     """
#     logging.info("Job started.")
#     start_time = datetime.now().strftime("%H:%M:%S.%f")
#     result_details = []

#     # Replace semicolons with spaces before executing
#     formatted_command = command.split(" ")
#     full_invocation = ['./isaaclab.sh', "-p"]
#     full_invocation.extend(formatted_command)
#     if test_mode:
#         # Use subprocess to call nvidia-smi and gather GPU info
#         logging.info("Checking GPUs.")
#         try:
#             result = subprocess.run(
#                 ["nvidia-smi", "--query-gpu=name,memory.free,serial", "--format=csv,noheader,nounits"],
#                 capture_output=True,
#                 check=True,
#                 text=True,
#             )
#             output = result.stdout.strip().split("\n")
#             for gpu_info in output:
#                 name, memory_free, serial = gpu_info.split(", ")
#                 result_details.append({"Name": name, "Memory Available": f"{memory_free} MB", "Serial Number": serial})
#         except subprocess.CalledProcessError as e:
#             logging.error(f"Error calling nvidia-smi: {e.stderr}")
#             result_details.append({"error": "Failed to retrieve GPU information"})
#     else:
#         try:
#             # Use subprocess.Popen for live output
#             process = subprocess.Popen(
#                 full_invocation, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
#             )

#             for line in process.stdout:
#                 print(line, end="")  # Print each line of the output live to the console
#                 logging.info(line.strip())  # Also log the line if needed

#             # Wait for the command to complete and get the rest of the output
#             stdout, stderr = process.communicate()

#             if stderr:
#                 logging.error(f"Error executing command: {stderr}")
#         except Exception as e:
#             logging.error(f"Exception occurred: {str(e)}")
#     num_gpus_detected = torch.cuda.device_count()
#     now = datetime.now().strftime("%H:%M:%S.%f")
#     result_str = f"Job Started at {start_time}, completed at {now} | "
#     result_str += f"# Detected GPUs: {num_gpus_detected} | Result details: {result_details}"
#     logging.info(result_str)  # Log the final result string
#     return result_str


# def main(num_workers, commands, gpus_per_worker, cpus_per_worker, memory_per_worker, test_mode):
#     ray.init(address="auto")
#     logging.info("Submitting jobs with specified resources to each worker...")

#     job_results = []
#     for i in range(num_workers):
#         command = commands[i] if not test_mode else "nvidia-smi"
#         logging.info(f"Submitting job {i + 1} with command '{command}'")
#         job = execute_command.options(
#             num_gpus=gpus_per_worker, num_cpus=cpus_per_worker, memory=memory_per_worker * 1024
#         ).remote(command, test_mode)
#         job_results.append(job)

#     results = ray.get(job_results)
#     for i, result in enumerate(results, 1):
#         print(f"Job {i} result: {result}")  # Print each job's result to the console

#     logging.info("All jobs completed.")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="Submit multiple jobs with optional GPU testing.")
    
#     # Adding cluster arguments from a utility or creating them manually if needed
#     isaac_ray_util.add_cluster_args(parser)  # Assumes this function adds cluster-related args
    
#     parser.add_argument(
#         "--commands",
#         nargs="+",
#         type=str,
#         help="List of commands to execute on workers, use ';' as delimiter between commands",
#     )
#     parser.add_argument("--test", action="store_true", help="Run nvidia-smi test instead of the arbitrary command")
    
#     # Parse known and unknown arguments
#     args, unknown = parser.parse_known_args()

#     # Ensure there are enough commands provided
#     if not args.test and len(args.commands) < args.num_workers:
#         logging.error("Not enough commands provided for the number of workers.")
#         exit(1)

#     # Calculate per-worker resource allocations
#     gpus_per_worker = args.cluster_gpu_count / args.num_workers
#     cpus_per_worker = args.cluster_cpu_count / args.num_workers
#     memory_per_worker = args.cluster_ram_gb / args.num_workers

#     # Append unknown arguments to commands
#     if args.commands:
#         full_jobs = [" ".join([job] + unknown) for job in args.commands]
#     else:
#         full_jobs = [" ".join(unknown)]  # If no jobs explicitly provided, take unknown args as the job command

#     # Run the main function with full job commands
#     main(args.num_workers, full_jobs, gpus_per_worker, cpus_per_worker, memory_per_worker, args.test)

import argparse
import logging
import subprocess
from datetime import datetime
import isaac_ray_util
import ray

# Set up logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


@ray.remote
def execute_command(command: str, test_mode: bool = False) -> str:
    import torch

    """
    Execute a specified command or check GPU availability if in test mode.
    Args:
        command (str): The command to execute.
        test_mode (bool): Flag to run GPU test with nvidia-smi.
    Returns:
        str: A description of the result or GPUs on the system.
    """
    logging.info("Job started.")
    start_time = datetime.now().strftime("%H:%M:%S.%f")
    result_details = []

    # Replace semicolons with spaces before executing
    formatted_command = command.split(";")
    full_invocation = ['./isaaclab.sh', "-p"]
    full_invocation.extend(formatted_command)
    if test_mode:
        # Use subprocess to call nvidia-smi and gather GPU info
        logging.info("Checking GPUs.")
        try:
            result = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,memory.free,serial", "--format=csv,noheader,nounits"],
                capture_output=True,
                check=True,
                text=True,
            )
            output = result.stdout.strip().split("\n")
            for gpu_info in output:
                name, memory_free, serial = gpu_info.split(", ")
                result_details.append({"Name": name, "Memory Available": f"{memory_free} MB", "Serial Number": serial})
        except subprocess.CalledProcessError as e:
            logging.error(f"Error calling nvidia-smi: {e.stderr}")
            result_details.append({"error": "Failed to retrieve GPU information"})
    else:
        try:
            # Use subprocess.Popen for live output
            process = subprocess.Popen(
                full_invocation, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
            )

            for line in process.stdout:
                print(line, end="")  # Print each line of the output live to the console
                logging.info(line.strip())  # Also log the line if needed

            # Wait for the command to complete and get the rest of the output
            stdout, stderr = process.communicate()

            if stderr:
                logging.error(f"Error executing command: {stderr}")
        except Exception as e:
            logging.error(f"Exception occurred: {str(e)}")
    num_gpus_detected = torch.cuda.device_count()
    now = datetime.now().strftime("%H:%M:%S.%f")
    result_str = f"Job Started at {start_time}, completed at {now} | "
    result_str += f"# Detected GPUs: {num_gpus_detected} | Result details: {result_details}"
    logging.info(result_str)  # Log the final result string
    return result_str


def main(num_workers, commands, gpus_per_worker, cpus_per_worker, memory_per_worker, test_mode):
    ray.init(address="auto")
    logging.info("Submitting jobs with specified resources to each worker...")

    job_results = []
    for i in range(num_workers):
        command = commands[i] if not test_mode else "nvidia-smi"
        logging.info(f"Submitting job {i + 1} with command '{command}'")
        job = execute_command.options(
            num_gpus=gpus_per_worker, num_cpus=cpus_per_worker, memory=memory_per_worker * 1024
        ).remote(command, test_mode)
        job_results.append(job)

    results = ray.get(job_results)
    for i, result in enumerate(results, 1):
        print(f"Job {i} result: {result}")  # Print each job's result to the console

    logging.info("All jobs completed.")


def split_commands(args):
    """
    Split commands and arguments by detecting .py scripts in the input args.
    Group all arguments with the .py script until the next .py script.
    """
    commands = []
    current_command = []
    
    for arg in args:
        if arg.endswith(".py"):
            if current_command:
                commands.append(" ".join(current_command))
            current_command = [arg]
        else:
            current_command.append(arg)
    
    # Add the final command
    if current_command:
        commands.append(" ".join(current_command))
    
    return commands


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Submit multiple jobs with optional GPU testing.")
    
    # Adding cluster arguments from a utility or creating them manually if needed
    isaac_ray_util.add_cluster_args(parser)  # Assumes this function adds cluster-related args
    
    parser.add_argument(
        "commands",
        nargs=argparse.REMAINDER,  # Capture all remaining arguments as commands
        help="Commands and their arguments to execute on workers.",
    )
    parser.add_argument("--test", action="store_true", help="Run nvidia-smi test instead of the arbitrary command")

    # Parse the arguments
    args = parser.parse_args()

    # Split commands and arguments by `.py`
    commands = split_commands(args.commands)

    # Calculate resource allocations
    gpus_per_worker = args.cluster_gpu_count / len(commands)
    cpus_per_worker = args.cluster_cpu_count / len(commands)
    memory_per_worker = args.cluster_ram_gb / len(commands)

    # Run the main function with full job commands
    main(len(commands), commands, gpus_per_worker, cpus_per_worker, memory_per_worker, args.test)
