import os

from .environment import TextCraftEnv
from .crafting_tree import CraftingTree
import threading

# Patched from upstream: the original default ("agentenv_textcraft/") is a
# relative path that only resolves if the process's CWD happens to be this
# package's own parent directory -- it isn't when imported from this repo's
# reward_hacking.envs.textcraft, which builds its own CraftingTree directly
# instead of using this module-level singleton anyway (see that module's
# docstring), but `import agentenv_textcraft` still runs this file's
# module-level `server = TextCraft_Wrapper()` regardless (Python always
# executes a package's __init__ chain, even for a submodule-only import), so
# it has to resolve correctly from any CWD.
_DEFAULT_MINECRAFT_DIR = os.path.dirname(os.path.abspath(__file__))


class TextCraft_Wrapper:
    def __init__(self, minecraft_dir=_DEFAULT_MINECRAFT_DIR):
        self._max_id = 0
        self.env = {}  # dict[id, env_item]
        self.info = {}  # dict[id, env_info]
        self.ls = []
        self.crafting_tree = CraftingTree(minecraft_dir=minecraft_dir)
        self._lock = threading.Lock()

    def create(self, commands: str = None, goal: str = None):
        try:
            with self._lock:
                id = self._max_id
                self._max_id += 1
            new_env = TextCraftEnv(
                crafting_tree=self.crafting_tree, commands=commands, goal=goal
            )
            ob, _ = new_env.reset(data_idx=id)
            print(f"-------Env {id} created--------")
            self.ls.append(id)
            payload = {"id": id, "observation": ob, "done": False, "reward": 0}
            self.env[id] = new_env
            self.info[id] = {
                "observation": ob,
                "done": False,
                "reward": 0,
                "deleted": False,
            }
        except Exception as e:
            payload = {"error": f"{e}"}
        return payload

    def step(self, id: int, action: str):
        try:
            self._check_id(id)
            (ob, reward, done, _, _) = self.env[id].step(action)
            payload = {"observation": ob, "reward": reward, "done": done}
            self.info[id].update(payload)
        except Exception as e:
            payload = {"error": f"{e}"}
        return payload

    def reset(self, id: int, data_idx: int):
        try:
            self._check_id(id)
            ob, _ = self.env[id].reset(data_idx=data_idx)
            payload = {"id": id, "observation": ob, "done": False, "reward": 0}
            self.info[id].update(
                {"observation": ob, "done": False, "reward": 0, "deleted": False}
            )
        except Exception as e:
            payload = {"error": str(e)}
        return payload

    def get_observation(self, id: int):
        try:
            self._check_id(id)
            return self.info[id]["observation"]
        except Exception as e:
            return {"error": str(e)}

    def get_detailed_info(self, id: int):
        try:
            self._check_id(id)
            return self.info[id]
        except Exception as e:
            return {"error": str(e)}

    def _check_id(self, id: int):
        if id not in self.info:
            raise NameError(f"The id {id} is not valid.")
        if self.info[id]["deleted"]:
            raise NameError(f"The task with environment {id} has been deleted.")
        
    def close(self, id):
        try:
            self.ls.remove(id)
            self.env[id].close() 
            del self.info[id] 
            del self.env[id] 
            print(f"-------Env {id} closed--------")
            return True
        except KeyError:
            print(f"--------Env {id} not exist--------")
            return False
        except Exception as e:
            print(f"Error while closing Env {id}: {e}")
            return False
    
    
    def __del__(self):
        for idx in self.ls:
            self.env[idx].close()
            print(f"-------Env {idx} closed--------")

server = TextCraft_Wrapper()
