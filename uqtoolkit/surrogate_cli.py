"""
Wrapper around argparse.ArgumentParser which automates some of the most 
frequently used arguments in the surrogate-related methods of uqtoolkit.
"""
import argparse
from os import makedirs
import hjson

class SurrogateCLI(argparse.ArgumentParser):
    def __init__(parser, unused=[], *args, **kwargs):
        super().__init__(*args, **kwargs)
        parser.add_argument(
            "--path-to-samples",
            default=".",
            type=str,
            help="/path/to/samples basedir",
        )
        parser.add_argument(
            "--uq-config",
            "-c",
            default=None,
            type=str,
            help="Name of config file used for setting up uq jobs",
        )
        parser.add_argument(
            "--pod-dir",
            default=None,
            type=str,
            help="/path/to/dir containing pod_coefs_sample*.txt",
        )
        parser.add_argument(
            "--plot-dir",
            default=None,
            type=str,
            help="/path/to/dir to output plots to",
        )
        parser.add_argument(
            "--num-modes",
            "-m",
            default=4,
            type=int,
            help="Number of POD modes to use",
        )
        for unused_i in unused:
            parser._remove_argument(unused_i)

    def parse_args(self, args=None, namespace=None):
        args, argv = self.parse_known_args(args, namespace)
        if argv:
            msg = ('unrecognized arguments: %s')
            self.error(msg % ' '.join(argv))
            
        # add default behaviour for pod-dir and plot-dir
        try:
            if args.pod_dir is None:
                args.pod_dir = f"{args.path_to_samples}/pod_data"
            makedirs(args.pod_dir, exist_ok=True)
        except AttributeError:
            pass

        try:
            # make plotting directory
            if args.plot_dir is None:
                args.plot_dir = f"{args.path_to_samples}/plots"
            makedirs(args.plot_dir, exist_ok=True)
        except AttributeError:
            pass

        try:
            if args.uq_config is not None:
                with open(args.uq_config) as f:
                    args.uq_config = hjson.load(f)
        except AttributeError:
            pass
        return args

    def _remove_argument(self, arg):
        """
        Private class which cycles through args in original object and compares
        against a list of strings for 'unused' args.
        The unused args are then removed from the object to avoid user confusion.
        """
        for action in self._actions:
            opts = action.option_strings
            if (opts and opts[0] == arg) or action.dest == arg:
                self._remove_action(action)
                break

        for action in self._action_groups:
            for group_action in action._group_actions:
                opts = group_action.option_strings
                if (opts and opts[0] == arg) or group_action.dest == arg:
                    action._group_actions.remove(group_action)
