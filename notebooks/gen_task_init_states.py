import numpy as np
import torch
import pickle
import os

from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

from IPython.display import display
from PIL import Image
from tqdm import tqdm

from libero.libero.benchmark import (
    register_benchmark,
    grab_language_from_filename,
    Benchmark,
    Task,
)

from libero.libero.benchmark.libero_suite_task_map import libero_task_map

"""
usage: 

from my_custom_benchmark_file import register_custom_libero

tasks = ["LIVING_ROOM_SCENE2_put_orange_juice_in_the_basket"]
suite_name = "libero_10_diff_obj"

# This will populate the task_map and register the benchmark
register_custom_libero(tasks, suite_name)

# Now you can use it like any other benchmark
from libero.libero import benchmark
b = benchmark.get_benchmark("libero_10_diff_obj")()
print(b.get_task_names())

"""


generate_pddl = False
generate_init_states = True
suite_name = "libero_10_cl2" # libero_10_train or libero_10_eval
overwrite = False

num_init_states = 100
num_pruned_states = 50

task_map = {}

# class LIBERO_10_diff_obj(Benchmark):
#     def __init__(self, task_order_index=0):
#         super().__init__(task_order_index=task_order_index)
#         self.name = "libero_10_diff_obj"
#         self._make_benchmark() 
    
#     def _make_benchmark(self):
#         tasks = list(task_map.values())
#         self.tasks = tasks
#         # not using task orders as of now (we want to be able to control the curriculum, but do this later )
#         # print(f"[info] using task orders {task_orders[self.task_order_index]}")
#         # self.tasks = [tasks[i] for i in task_orders[self.task_order_index]]
#         self.n_tasks = len(self.tasks)


def register_custom_libero(suite_name):
    """Register a new benchmark and optionally generate init states."""

    assert suite_name in libero_task_map, f"Could not find custom suite {suite_name} in libero_task_map.py"
    tasks = libero_task_map[suite_name]
    
    global task_map
    task_map = {}

    init_save_dir = get_libero_path("init_states")
    
    for task in tqdm(tasks, "All tasks", position=0):
        language = grab_language_from_filename(task + ".bddl")
        print(language)
        task_map[task] = Task(
            name=task,
            language=language,
            problem="Libero",
            problem_folder=suite_name,
            bddl_file=f"{task}.bddl",
            init_states_file=f"{task}.pruned_init",
        )

        # print(f"{os.path.join(init_save_dir, suite_name, task + '.init')} exists?: {os.path.exists(os.path.join(init_save_dir, suite_name, task + '.init'))}")
        # generate init states if it doesn't exist
        if overwrite:
            print(f"Resampling init file for {os.path.join(init_save_dir, suite_name, task + '.init')} now ...")
            generate_init_states_for_task(task, suite_name, num_init_states, num_pruned_states)
        elif not os.path.exists(os.path.join(init_save_dir, suite_name, task + ".init")):
            print(f"path doesn't exist: {os.path.join(init_save_dir, suite_name, task + '.init')}, generating init file now ...")
            generate_init_states_for_task(task, suite_name, num_init_states, num_pruned_states)
    
    # Register benchmark class
    # register_benchmark(LIBERO_10_diff_obj)



# TODO: add this part of the code if necessary
######## Generate PDDL file ###########

# Currently just changing objects that are manipulated, we can just copy the previous pddl files




######## Generate Init States #########

# Currenty just changing objects that are manipulated, we can just copy the previous init files
def generate_init_states_for_task(task, suite_name, num_init_states=100, num_pruned_states=50):
    # TODO: allow multiiple tasks to be generated
    # task = "LIVING_ROOM_SCENE2_put_orange_juice_in_the_basket"
    env_args = {
        "bddl_file_name": os.path.join(get_libero_path("bddl_files"), suite_name,  task + ".bddl"),
        "camera_heights": 256,
        "camera_widths": 256
    }

    env = OffScreenRenderEnv(**env_args)

    init_states = []
    for i in tqdm(range(num_init_states), desc=f"Task: {task}\n"[:75], position=1, leave=False):
        obs = env.reset()
        init_state = env.sim.get_state().flatten()
        init_states.append(init_state)
    
    init_states = np.array(init_states)
    pruned_states = init_states[:num_pruned_states]

    # data = pickle.dumps(init_states)
    # pruned_data = pickle.dumps(pruned_states)

    # Save in LIBERO format (npz container with 'archive/data.pkl')
    # archive = {
    #     "archive/data.pkl": np.array(data),
    #     "archive/version": b"1\n", # this is not exact format because of np.savez wraps in np.array
    # }

    # pruned_archive = {
    #     "archive/data.pkl": np.array(pruned_data),
    #     "archive/version": b"1\n",
    # }

    # TODO: change to be for save directory and automatically
    save_dir = f"/home/whu/LIBERO_PLUS/libero/libero/init_files/{suite_name}"
    # save_dir = f"/home/leisongao/vlarl/LIBERO/libero/libero/./init_files/{suite_name}"
    
    os.makedirs(save_dir, exist_ok=True)

    save_path = os.path.join(save_dir, task)

    # torch.save(np.array(data), save_path + ".init")
    # torch.save(np.array(pruned_data), save_path + ".pruned_init")
    

    torch.save(init_states, save_path + ".init")
    torch.save(pruned_states, save_path + ".pruned_init")
    

    print(f"Saved init files in {save_path}")



register_custom_libero(suite_name)