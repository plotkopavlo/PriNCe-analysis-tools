template_submit = """#!/bin/zsh

#singularity run /project/singularity/images/SL7.img
source .zshenv
source .zshrc

#$ -N {project_tag}
#$ -l h_rt=1:00:00
#$ -l h_rss=2G
#$ -j y
#$ -m ae
#$ -l os=sl7
#$ -o {folder_log}/{project_tag}$TASK_ID.log

OUTFILE={folder_out}/{project_tag}$SGE_TASK_ID.out
TMPOUT=$TMPDIR/tmp.out

echo Starting job with options on
echo `hostname`. Now is `date`

python {runfile} -r --jobid $SGE_TASK_ID --outfile $TMPOUT
#singularity exec /project/singularity/images/SL7.img python {runfile} -r --jobid $SGE_TASK_ID --outfile $TMPOUT

#Copy output to destination
mv $TMPOUT $OUTFILE
"""


class PropagationProject(object):
    def __init__(self, conf, dryrun=False):
        import os.path as path
        self.conf = conf

        # tag used for filenames and basefolder
        self.project_tag = conf['project_tag']
        # basefolder where to create the project tag
        self.targetdir = path.join(conf['targetdir'], self.project_tag)
        # path where the input file is located
        self.inputpath = conf['inputpath']

        # the subfolders, log files and output
        self.folder_log = path.join(self.targetdir, 'log')
        self.folder_out = path.join(self.targetdir, 'out')
        self.runfile = path.join(self.targetdir,
                                 'run_' + self.project_tag + '.py')
        self.subfile = path.join(self.targetdir,
                                 'sub_' + self.project_tag + '.sh')
        # list of parameters to run the prog with
        self.paramlist = conf['paramlist']
        self.njobs = conf['njobs']

    def setup_project(self):
        """Sets up the standard folders and files in the project folder"""
        from os import makedirs, path

        # step 1: create the project folders
        try:
            makedirs(self.folder_log)
            makedirs(self.folder_out)
        except:
            raise Exception(
                "_setup_project():: folders already exists, delete old files first!!"
            )

        # step 2: create files in the target folder
        from shutil import copyfile
        copyfile(self.inputpath, self.runfile)

        # step 3: create a submit file from template
        with open(self.subfile, 'w') as subfile:
            subfile.write(
                template_submit.format(
                    project_tag=self.project_tag,
                    runfile=self.runfile,
                    folder_log=self.folder_log,
                    folder_out=self.folder_out,
                ))

    def scan_logfiles(self):
        """Scans the log folder for missing files"""
        import os
        import re
        expected = range(1, self.njobs + 1)
        found = [
            idx for idx in expected
            if self.project_tag + '{:}.log'.format(idx) in os.listdir(
                self.folder_log)
        ]
        missing = [
            idx for idx in expected
            if self.project_tag + '{:}.log'.format(idx) not in os.listdir(
                self.folder_log)
        ]
        print 'found logfiles:'
        print found
        print 'missing logfiles:'
        print missing
        return found, missing

    def scan_output(self):
        """Scans the output folder for missing files"""
        import os
        import re
        expected = range(1, self.njobs + 1)
        found = [
            idx for idx in expected
            if self.project_tag + '{:}.out'.format(idx) in os.listdir(
                self.folder_out)
        ]
        missing = [
            idx for idx in expected
            if self.project_tag + '{:}.out'.format(idx) not in os.listdir(
                self.folder_out)
        ]
        print 'found outputfiles:'
        print found
        print 'missing outputfiles:'
        print missing
        return found, missing

    def submit_all_jobs(self):
        """Submits a job array"""
        from os import listdir, path, chdir
        import subprocess
        retcode = subprocess.call(
            ['qsub', '-t', '1:{:}'.format(self.njobs), self.subfile])

    def run_subset(self, jobid, outputfile):
        """Run the calculations for a subset of the parameter space"""
        import numpy as np
        import itertools as it

        # Create a list of all permutations of the scan parameters
        ranges = self.paramlist
        permutations = it.product(
            *[range(arr.size) for arr in ranges.values()])
        permutations = list(permutations)

        # Runs the function supplied by config on a a fraction of the parameter space
        # Fraction depends on the number of total jobs
        setup = self.conf['setup_func']()
        results = []
        for perm in permutations[jobid - 1::self.njobs]:
            inp = {}
            for (key, arr), idx in zip(ranges.items(), perm):
                inp[key] = arr[idx]

            func = self.conf['single_run_func']
            results.append(func(setup, **inp))

        # Save the list of results to pickle
        import cPickle as pickle
        with open(outputfile, "wb") as thefile:
            pickle.dump(results, thefile)

    def submit_missing_jobs(self):
        pass

    def collect_job_results(self):
        """Collect the computed results to a single array"""
        _, missing = self.scan_output()
        if len(missing) >= 0:
            raise Exception(
                'Cannot collect results, not all results were computed yet!')

        import numpy as np
        import itertools as it
        import cPickle as pickle
        import os.path as path

        # Create a list of all permutations of the scan parameters
        ranges = self.paramlist
        permutations = it.product(
            *[range(arr.size) for arr in ranges.values()])
        permutations = list(permutations)

        # Create an array of the needed size
        shape = (arr.size for arr in ranges.values)
        collected = np.zeros(shape, dtype=object)

        # Loop over the single output files
        for jobid in range(1, self.njobs + 1):
            outputfile = path.join(self.folder_out,
                                   self.project_tag + '{:}.out'.format(jobid))
            with open(outputfile, "rb") as thefile:
                results = pickle.load(thefile)

            for res, perm in zip(results, permutations[jobid - 1::self.njobs]):
                collected[perm] = res

        # Save the list of results to pickle
        import cPickle as pickle
        outputfile = path.join(self.targetdir,
                               'collected_' + self.project_tag + '.out')
        with open(outputfile, "wb") as thefile:
            pickle.dump((ranges, collected), thefile)

    def run_from_terminal(self):
        from optparse import OptionParser, OptionGroup
        usage = \
        """usage: %prog [options] args"""
        parser = OptionParser(usage=usage)

        parser.add_option(
            '-c',
            '--create',
            dest='create',
            action='store_true',
            help='If this is set, the project folder will be created')
        parser.add_option(
            '-s',
            '--submit',
            dest='submit',
            action='store_true',
            help='If this is set, the project will be submitted to the cluster'
        )
        parser.add_option(
            '-m',
            '--miss',
            dest='missing',
            action='store_true',
            help=
            'If this is set, the project search for missing output and logfiles'
        )
        parser.add_option(
            '-r',
            '--run',
            dest='run',
            action='store_true',
            help=
            'If this is set, a single calculations from the project will be run'
        )
        parser.add_option(
            '--collect',
            dest='collect',
            action='store_true',
            help=
            'If this is set, the project results will be collected into a single folder'
        )

        run_group = OptionGroup(
            parser, "Options for a single calculations, need -r to be set")
        run_group.add_option(
            "--outfile",
            dest="outputfile",
            type="str",
            help="ouput will be written to this file")
        run_group.add_option(
            "--jobid",
            dest="jobid",
            type="int",
            help="ID of the job in distributed scan")
        parser.add_option_group(run_group)
        options, args = parser.parse_args()

        if options.create:
            self.setup_project()
        elif options.submit:
            self.submit_all_jobs()
        elif options.missing:
            self.scan_logfiles()
            self.scan_output()
        elif options.run:
            self.run_subset(options.jobid, options.outputfile)
        elif options.collect:
            self.collect_job_results()
        else:
            raise Exception('No valid options specified, set either -s -r -c')
