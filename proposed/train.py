from argparse import ArgumentParser
import yaml
from solver import Solver

def main():
    parser = ArgumentParser()
    parser.add_argument('-config', '-c', default='config.yaml')
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    print(yaml.dump(config, default_flow_style=False))

    solver = Solver(config)
    solver.train()

if __name__ == "__main__":
    main()
