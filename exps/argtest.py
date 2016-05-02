import argparse

parser = argparse.ArgumentParser()


parser.add_argument('--feature',action='store_true')

args = parser.parse_args()

print args.feature