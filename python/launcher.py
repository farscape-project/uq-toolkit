from os import path

class UQLauncher:
    def __init__(self, launcher_type, launcher_script):
        self.launcher_type = launcher_type
        if self.launcher_type == "None":
            self.launcher_type = None
        self.launch_string = self._get_launcher_string()
        if self.launcher_type is not None:
            assert path.exists(f"basedir/{launcher_script}"), "template launcher script does not exist"
        self.launcher_script = launcher_script
        self.scheduler_text = self._setup_scheduler()

    def _get_launcher_string(self):
        if self.launcher_type == "bash":
            return "bash "
        elif self.launcher_type == "slurm":
            return "sbatch "
        elif self.launcher_type == "lsf":
            return "bsub < "
        elif self.launcher_type == None:
            return
        else:
            raise NotImplementedError

    def _setup_scheduler(self):
        # TODO: Add slurm and others
        scheduler_text = []
        if self.launcher_type in ["bash", "slurm"]:
            # note that when using mpi, each simulation is run in parallel, but the batch itself is run in serial
            scheduler_text.append("#!/bin/bash\n\n")
            pass
        elif self.launcher_type == None:
            return
        else:
            raise NotImplementedError
        return scheduler_text

    def append_to_scheduler(self, newdir):
        if self.launcher_type in ["bash", "slurm"]:
            self.scheduler_text.append(f"cd {newdir}\n")
            self.scheduler_text.append(f"{self.launch_string} {self.launcher_script} \n")
            self.scheduler_text.append(f"cd -\n")

    def write_launcher(self, launcher_name):
        if self.launcher_type in ["bash", "slurm"]:
            self._write_bash_launcher(launcher_name)

    def _write_bash_launcher(self, launcher_name):
        with open(launcher_name, "w") as f:
            [f.write(line_i) for line_i in self.scheduler_text]
