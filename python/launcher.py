from os import path


class UQLauncher:
    def __init__(self, launcher_type, launcher_script, launcher_dir):
        self.launcher_type = launcher_type
        if self.launcher_type == "None":
            self.launcher_type = None
        self.launch_string = self._get_launcher_string()
        if self.launcher_type is not None:
            expected_path = f"{launcher_dir}/{launcher_script}"
            assert path.exists(expected_path), (
                "template launcher script does not exist \n"
                f"check this path is correct: {expected_path}"
            )
        self.launcher_script = launcher_script
        self.scheduler_text = self._setup_scheduler()

    def _get_launcher_string(self):
        if self.launcher_type == "bash":
            return "bash "
        elif self.launcher_type == "slurm":
            return "sbatch "
        elif self.launcher_type == "lsf":
            return "bsub "
        elif self.launcher_type == None:
            return
        else:
            raise NotImplementedError

    def _setup_scheduler(self):
        scheduler_text = []
        if self.launcher_type in ["bash", "slurm", "lsf"]:
            # note that when using mpi, each simulation is run in parallel, but the batch itself is run in serial
            scheduler_text.append("#!/bin/bash\n\n")
            pass
        elif self.launcher_type == None:
            return
        else:
            raise NotImplementedError
        return scheduler_text

    def append_to_scheduler(self, newdir, job_name):
        if self.launcher_type in ["bash", "slurm", "lsf"]:
            self.scheduler_text.append(f"cd {newdir}\n")
            # lsf has extra "<" char before script name
            script_char = ""
            if self.launcher_type == "lsf":
                script_char = "<"
            self.scheduler_text.append(
                f"{self.launch_string} -J {job_name} {script_char} {self.launcher_script} \n"
            )
            self.scheduler_text.append(f"cd -\n")

    def write_launcher(self, launcher_name):
        if self.launcher_type in ["bash", "slurm", "lsf"]:
            self._write_bash_launcher(launcher_name)

    def _write_bash_launcher(self, launcher_name):
        with open(launcher_name, "w") as f:
            [f.write(line_i) for line_i in self.scheduler_text]
