import time
import os

# run all classes

EPOCHS = 30
LAMBDA = 10
WORKERS = 10


def run_experiments(classes, lambdas, examples_options):
    i = 0
    for cls in classes:
        for lamb in lambdas:
            for num_examples in examples_options:
                command = f'python run.py  --rel-class {cls} --lamb {lamb} --num-examples {num_examples}'
                if i % 2 == 0:
                    # run without waiting
                    os.system(command + " --gpu-index 0 &")
                else:
                    # run on second GPU and wait for both
                    os.system(command + ' --gpu-index 1')
                    # extra 5 minutes to be sure both finished
                    time.sleep(60 * 5)
                i += 1


if __name__ == '__main__':
    # all classes, fixed lamb +
    run_experiments(list(range(10)), [10], [250])
    # lambda vals
    run_experiments([0], [1, 2, 5, 10, 20, 50], [250])
    # different num samples
    run_experiments([0], [10], [10, 50, 100, 250, 500])
